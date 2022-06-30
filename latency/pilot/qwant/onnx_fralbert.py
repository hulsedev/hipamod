import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, pipeline, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from optimum.onnxruntime.modeling_ort import ORTModel, ORTModelForFeatureExtraction

from latency import utils
from latency.pilot.qwant import benchmark_fralbert


class ORTModelForMaskedLM(ORTModel):
    """
    Fill Mask model for ONNX.
    """

    # used in from_transformers to export model to onnx
    pipeline_task = "fill-mask"
    auto_model_class = AutoModelForMaskedLM

    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)
        # create {name:idx} dict for model outputs
        self._device = self.get_device_for_provider(self.model.get_providers()[0])
        self.model_outputs = {
            output_key.name: idx
            for idx, output_key in enumerate(self.model.get_outputs())
        }
        self.model_inputs = {
            output_key.name: idx
            for idx, output_key in enumerate(self.model.get_inputs())
        }

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return self._device

    def get_device_for_provider(self, provider: str) -> torch.device:
        """
        Gets the PyTorch device (CPU/CUDA) associated with an ONNX Runtime provider.
        """
        return (
            torch.device("cuda")
            if provider == "CUDAExecutionProvider"
            else torch.device("cpu")
        )

    @device.setter
    def device(self, value):
        self._device = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }

        if token_type_ids is not None:
            onnx_inputs["token_type_ids"] = token_type_ids.cpu().detach().numpy()
        # run inference
        outputs = self.model.run(None, onnx_inputs)
        logits = torch.from_numpy(outputs[self.model_outputs["logits"]]).to(self.device)
        # converts output to namedtuple for pipelines post-processing
        return MaskedLMOutput(
            logits=logits,
        )


def main():
    """Benchmark end to end compute latency with ONNX model."""
    tokenizer = AutoTokenizer.from_pretrained("qwant/fralbert-base")
    model_filepath = Path("fralbert_base.onnx")
    output_dir = Path("latency/pilot/qwant/model/")
    onnx_model = ORTModelForMaskedLM.from_pretrained(
        output_dir, file_name=model_filepath
    )
    unmasker = pipeline("fill-mask", model=onnx_model, tokenizer=tokenizer)

    compute_latency = 0
    for _ in range(utils.random_folds):
        start = time.time()
        output = unmasker(benchmark_fralbert.text)
        compute_latency += time.time() - start

    print(f"Compute latency: {compute_latency / utils.random_folds * 1000:.2f}ms")


if __name__ == "__main__":
    main()

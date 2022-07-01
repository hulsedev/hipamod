from pathlib import Path
from typing import Mapping, List, Optional, Any
from collections import OrderedDict
from copy import copy
from typing import Union, IO
import os

import onnx
from transformers.onnx import (
    export,
    validate_model_outputs,
    OnnxConfigWithPast,
    PatchingSpec,
)
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    is_torch_available,
    TensorType,
    AutoModelForCausalLM,
)
from transformers.configuration_utils import PretrainedConfig

# from latency.pilot.opt.compress.utils import OPTOnnxConfig


class OPTOnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        super().__init__(
            config, task=task, patching_specs=patching_specs, use_past=use_past
        )
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {
                0: "batch",
                1: "past_sequence + sequence",
            }
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    # @property
    # def outputs(self) -> Mapping[str, Mapping[int, str]]:
    #    # we know that task is "causal-lm"
    #    outputs = OrderedDict({"loss": {0: "batch", 1: "sequence"}})
    #    outputs["logits"] = {0: "batch", 1: "sequence"}
    #    return outputs

    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer,
            batch_size=batch_size,
            seq_length=seq_length,
            is_pair=is_pair,
            framework=framework,
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError(
                    "Cannot generate dummy past_keys inputs without PyTorch installed."
                )
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape))
                    for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [
                    ordered_inputs["attention_mask"],
                    torch.ones(batch, past_key_values_length, dtype=mask_dtype),
                ],
                dim=1,
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13


def load_model(
    model_dir: Union[str, Path],
    f: Union[IO[bytes], str],
    format: Optional[Any] = None,
    load_external_data: bool = True,
):
    s = onnx._load_bytes(f)
    model = onnx.load_model_from_string(s, format=format)

    if load_external_data:
        model_filepath = onnx._get_file_path(f)
        if model_filepath:
            base_dir = model_dir  # os.path.dirname(model_filepath)
            onnx.load_external_data_for_model(model, base_dir)

    return model


def main(model_name):
    # shooting for 5ms compute latency
    print(f"Exporting {model_name} to ONNX")
    model_dir = Path(f"latency/pilot/opt/model/{model_name}/")
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    model_ckpt = f"facebook/{model_name}"
    # override definition of base model too
    base_model = AutoModelForCausalLM.from_pretrained(model_ckpt)
    onnx_config = OPTOnnxConfig(base_model.config, task="causal-lm")

    onnx_path = model_dir.joinpath("model.onnx")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    onnx_inputs, onnx_outputs = export(
        tokenizer, base_model, onnx_config, onnx_config.default_onnx_opset, onnx_path
    )

    onnx_model = load_model(model_dir, onnx_path)
    onnx.checker.check_model(onnx_model)
    validate_model_outputs(
        onnx_config,
        tokenizer,
        base_model,
        onnx_path,
        onnx_outputs,
        1e-4,  # onnx_config.atol_for_validation,
    )
    print(
        f"Successfully exported {model_ckpt} to ONNX format. Available at {onnx_path}"
    )


if __name__ == "__main__":
    for model_name in [
        # "opt-125m",
        "opt-350m",
        # "opt-1.3b",
        # "opt-2.7b",
        # "opt-6.7b",
        # "opt-13b",
        # "opt-30b",
    ]:
        main(model_name)

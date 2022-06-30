from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import onnx
from transformers import AlbertForMaskedLM, AutoConfig, AutoModel, AutoTokenizer
from transformers.onnx import OnnxConfig, export, validate_model_outputs


class FrAlbertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )


def main():
    model_ckpt = "qwant/fralbert-base"
    base_model = AlbertForMaskedLM.from_pretrained(model_ckpt)
    fralbert_config = base_model.config
    onnx_config = FrAlbertOnnxConfig(config=fralbert_config, task="masked-lm")
    print("Checking onnx operator set", onnx_config.default_onnx_opset)
    onnx_path = Path("latency/pilot/qwant/model/fralbert_base.onnx")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    onnx_inputs, onnx_outputs = export(
        tokenizer, base_model, onnx_config, onnx_config.default_onnx_opset, onnx_path
    )

    # check that model format makes sense
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # check that outputs do match the current ones
    validate_model_outputs(
        onnx_config,
        tokenizer,
        base_model,
        onnx_path,
        onnx_outputs,
        1e-4,  # onnx_config.atol_for_validation,
    )
    print(
        "Successfully exported model to ONNX format. Available at {}".format(onnx_path)
    )


if __name__ == "__main__":
    main()

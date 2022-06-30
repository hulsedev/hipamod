from pathlib import Path
from typing import Mapping
from collections import OrderedDict

import onnx
from transformers.onnx import export, validate_model_outputs
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
)
from transformers.models.distilbert.configuration_distilbert import DistilBertOnnxConfig

from latency import utils


def main():
    # shooting for 5ms compute latency
    model_dir = Path("latency/pilot/distilbert_base_sst/model/")
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    base_model = DistilBertForSequenceClassification.from_pretrained(model_ckpt)
    onnx_config = DistilBertOnnxConfig(
        base_model.config, task="sequence-classification"
    )
    onnx_path = model_dir.joinpath("model.onnx")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    onnx_inputs, onnx_outputs = export(
        tokenizer, base_model, onnx_config, onnx_config.default_onnx_opset, onnx_path
    )

    onnx_model = onnx.load(onnx_path)
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
        "Successfully exported model to ONNX format. Available at {}".format(onnx_path)
    )


if __name__ == "__main__":
    main()

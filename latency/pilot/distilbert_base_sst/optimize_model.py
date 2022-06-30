from pathlib import Path
import sys

from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig


def main():
    save_path = Path("latency/pilot/distilbert_base_sst/model/")
    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    optimization_config = OptimizationConfig(optimization_level=99)
    optimizer = ORTOptimizer.from_pretrained(
        model_ckpt, feature="sequence-classification"
    )
    base_model_filename = "model.onnx"

    optimizer.export(
        onnx_model_path=save_path / base_model_filename,
        onnx_optimized_model_output_path=save_path / "model_optimized.onnx",
        optimization_config=optimization_config,
    )
    optimizer.model.config.save_pretrained(save_path)


if __name__ == "__main__":
    main()

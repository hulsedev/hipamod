from pathlib import Path
import sys

from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig


def main(base_model_type):
    save_path = Path("latency/pilot/qwant/model/")
    model_ckpt = "qwant/fralbert-base"
    optimization_config = OptimizationConfig(optimization_level=99)
    optimizer = ORTOptimizer.from_pretrained(model_ckpt, feature="masked-lm")

    if base_model_type == "quantized":
        base_model_filename = "fralbert_base_quantized.onnx"
    else:
        base_model_filename = "fralbert_base.onnx"

    optimizer.export(
        onnx_model_path=save_path / base_model_filename,
        onnx_optimized_model_output_path=save_path / "fralbert_base_optimized.onnx",
        optimization_config=optimization_config,
    )
    optimizer.model.config.save_pretrained(save_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_model_type = sys.argv[1]
    else:
        base_model_type = "unquantized"
    main(base_model_type)

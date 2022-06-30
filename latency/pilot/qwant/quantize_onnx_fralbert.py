import sys
from pathlib import Path

from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer, ORTModel
from transformers import AutoTokenizer, pipeline


from latency.pilot.qwant import benchmark_fralbert
from latency.pilot.qwant.onnx_fralbert import ORTModelForMaskedLM


def main(base_model_type):
    model_ckpt = "qwant/fralbert-base"
    output_dir = Path("latency/pilot/qwant/model/")
    if base_model_type == "optimized":
        print("Quantizing optimized model")
        onnx_path = Path("latency/pilot/qwant/model/fralbert_base_optimized.onnx")
    else:
        print("Quantizing non-optimized model")
        onnx_path = Path("latency/pilot/qwant/model/fralbert_base.onnx")
    output_file = Path("fralbert_base_quantized.onnx")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    quantizer = ORTQuantizer.from_pretrained(model_ckpt, feature="masked-lm")
    qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=True)
    quantizer.export(
        onnx_model_path=onnx_path,
        onnx_quantized_model_output_path=output_dir.joinpath(output_file),
        quantization_config=qconfig,
    )
    quantizer.model.config.save_pretrained(output_dir)

    model = ORTModelForMaskedLM.from_pretrained(output_dir, file_name=output_file)
    onnx_clx = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    pred = onnx_clx(benchmark_fralbert.text)
    print("Done quantizing model.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_model_type = sys.argv[1]
    else:
        base_model_type = "unoptimized"
    main(base_model_type)

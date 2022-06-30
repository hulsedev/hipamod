import sys
from pathlib import Path

from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer, ORTModel
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime.modeling_ort import ORTModelForSequenceClassification


from latency.pilot.qwant import benchmark_fralbert


def main():
    model_dir = Path("latency/pilot/distilbert_base_sst/model/")
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    onnx_path = Path("latency/pilot/qwant/model/model.onnx")
    output_file = Path("model_quantized.onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    quantizer = ORTQuantizer.from_pretrained(
        model_ckpt, feature="sequence-classification"
    )
    qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)
    quantizer.export(
        onnx_model_path=onnx_path,
        onnx_quantized_model_output_path=model_dir.joinpath(output_file),
        quantization_config=qconfig,
    )
    quantizer.model.config.save_pretrained(model_dir)
    print("Done quantizing model.")


if __name__ == "__main__":
    main()

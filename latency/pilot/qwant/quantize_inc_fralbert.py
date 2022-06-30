import sys
from pathlib import Path

from optimum.intel import IncQuantizer, IncConfig
from transformers import AutoTokenizer, pipeline


from latency.pilot.qwant import benchmark_fralbert
from latency.pilot.qwant.onnx_fralbert import ORTModelForMaskedLM


def main():
    # TODO: double check if still possible later
    model_ckpt = "qwant/fralbert-base"
    output_dir = Path("latency/pilot/qwant/model/")
    if base_model_type == "optimized":
        print("Quantizing optimized model")
        onnx_path = Path("latency/pilot/qwant/model/fralbert_base_inc_optimized.onnx")
    else:
        print("Quantizing non-optimized model")
        onnx_path = Path("latency/pilot/qwant/model/fralbert_base.onnx")

    output_file = Path("fralbert_base_inc_quantized.onnx")

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = ORTModelForMaskedLM.from_pretrained(
        output_dir, file_name=onnx_path.filename
    )
    qconfig = IncConfig.from_pretrained(output_dir)

    quantizer = IncQuantizer(model=model, tokenizer=tokenizer)
    quantizer.export(
        onnx_model_path=onnx_path,
        onnx_quantized_model_output_path=output_dir.joinpath(output_file),
        quantization_config=qconfig,
    )

    quantizer.model.config.save_pretrained(output_dir)
    print("Done quantizing model using inc.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_model_type = sys.argv[1]
    else:
        base_model_type = "unoptimized"
    main()

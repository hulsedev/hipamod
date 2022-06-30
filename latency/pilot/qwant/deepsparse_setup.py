from pathlib import Path

from transformers import AutoTokenizer

from latency.pilot.qwant.onnx_fralbert import ORTModelForMaskedLM


def main():
    """Prepare all model and tokenizer checkpoints for deepsparse inference."""
    current_dir = Path("latency/pilot/qwant/model/")
    output_dir = Path("latency/pilot/qwant/sparse/")
    output_dir.mkdir(parents=True)
    model_ckpt = "qwant/fralbert-base"
    filename = "fralbert_base_optimized.onnx"

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = ORTModelForMaskedLM.from_pretrained(current_dir, file_name=filename)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

from pathlib import Path
import sys

from deepsparse import compile_model
from transformers import AutoTokenizer
from optimum.onnxruntime.modeling_ort import ORTModelForCausalLM


def setup(model_name):
    current_dir = Path(f"latency/pilot/opt/model/{model_name}/")
    output_dir = Path(f"latency/pilot/opt/sparse/{model_name}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt = f"facebook/{model_name}"
    filename = "model.onnx"

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = ORTModelForCausalLM.from_pretrained(current_dir, file_name=filename)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


def main(model_name):
    output_dir = Path(f"latency/pilot/opt/sparse/{model_name}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt = f"facebook/{model_name}"
    filename = "model.onnx"
    model_path = output_dir.joinpath(filename)


if __name__ == "__main__":
    model_name = "opt-350m"

    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup(model_name)
    else:
        main(model_name)

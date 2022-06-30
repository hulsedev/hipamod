import time
from pathlib import Path

from transformers import pipeline
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime.modeling_ort import ORTModelForCausalLM

from latency import utils


def main():
    model_dir = Path("latency/pilot/opt/model/")
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    model_ckpt = "facebook/opt-350m"
    onnx_model = ORTModelForCausalLM.from_pretrained(
        model_dir, file_name="model_quantized.onnx"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    classifier = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)

    compute_latency = []
    for i in tqdm(range(utils.random_folds)):
        start = time.time()
        _ = classifier(utils.prompts.get("8")[i % utils.random_folds])
        compute_latency.append(time.time() - start)

    print(
        f"Compute latency: {np.mean(compute_latency) * 1000:.2f}ms - std {np.std(compute_latency) * 1000:.2f}ms"
    )


if __name__ == "__main__":
    main()
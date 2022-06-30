from pathlib import Path
import time

from transformers import pipeline, AutoTokenizer
import numpy as np
from tqdm import tqdm

from latency.pilot.qwant.onnx_fralbert import ORTModelForMaskedLM
from latency.pilot.qwant import benchmark_fralbert
from latency import utils


def main():
    save_path = Path("latency/pilot/qwant/model/")
    model_ckpt = "qwant/fralbert-base"
    filename = "fralbert_base_optimized.onnx"

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = ORTModelForMaskedLM.from_pretrained(save_path, file_name=filename)
    onnx_clx = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    compute_latency = []
    for i in tqdm(range(utils.random_folds * 10)):
        prompt = utils.prompts.get("8")[i % utils.random_folds].split()
        prompt[4] = "[MASK]"
        prompt = " ".join(prompt)
        start = time.time()
        _ = onnx_clx(prompt)
        compute_latency.append(time.time() - start)

    print(
        f"Compute latency: {np.mean(compute_latency) * 1000:.2f}ms - std {np.std(compute_latency) * 1000:.2f}ms"
    )


if __name__ == "__main__":
    main()

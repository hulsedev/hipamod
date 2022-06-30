from pathlib import Path
import time

from transformers import pipeline, AutoTokenizer

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

    compute_latency = 0
    for _ in range(utils.random_folds):
        start = time.time()
        output = onnx_clx(benchmark_fralbert.text)
        compute_latency += time.time() - start

    print(f"Compute latency: {compute_latency / utils.random_folds * 1000:.2f}ms")


if __name__ == "__main__":
    main()

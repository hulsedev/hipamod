import time
from pathlib import Path

from transformers import pipeline, AutoTokenizer

from latency.pilot.qwant.onnx_fralbert import ORTModelForMaskedLM
from latency.pilot.qwant import benchmark_fralbert
from latency import utils


def main():
    output_dir = Path("latency/pilot/qwant/model/")
    output_file = Path("fralbert_base_quantized.onnx")

    tokenizer = AutoTokenizer.from_pretrained("qwant/fralbert-base")
    model = ORTModelForMaskedLM.from_pretrained(output_dir, file_name=output_file)
    onnx_clx = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    compute_latency = 0
    for _ in range(utils.random_folds):
        start = time.time()
        output = onnx_clx(benchmark_fralbert.text)
        compute_latency += time.time() - start

    print(f"Compute latency: {compute_latency / utils.random_folds * 1000:.2f}ms")


if __name__ == "__main__":
    main()

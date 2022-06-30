from pathlib import Path
import time

from deepsparse import compile_model
from deepsparse.transformers import pipeline
from deepsparse.utils import generate_random_inputs

from latency import utils
from latency.pilot.qwant import benchmark_fralbert


def main():
    model_path = Path("latency/pilot/qwant/sparse/")
    classifier = pipeline(
        task="text-classification",
        model_path=model_path,
    )
    # engine = compile_model(str(model_path), 1)

    compute_latency = 0
    for _ in range(utils.random_folds):
        start = time.time()
        output = classifier(benchmark_fralbert.text)
        # _ = engine.run(benchmark_fralbert.text)
        compute_latency += time.time() - start

    print(f"Compute latency: {compute_latency / utils.random_folds * 1000:.2f}ms")


if __name__ == "__main__":
    main()

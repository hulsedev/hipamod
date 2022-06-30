import time

from transformers import pipeline
from tqdm import tqdm
import numpy as np

from latency import utils


def main():
    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    classifier = pipeline("text-classification", model=model_ckpt)

    compute_latency = []
    for i in tqdm(range(utils.random_folds * 10)):
        start = time.time()
        _ = classifier(utils.prompts.get("8")[i % utils.random_folds])
        compute_latency.append(time.time() - start)

    print(
        f"Compute latency: {np.mean(compute_latency) * 1000:.2f}ms - std {np.std(compute_latency) * 1000:.2f}ms"
    )


if __name__ == "__main__":
    main()

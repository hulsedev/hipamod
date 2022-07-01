import time
import random

random.seed(42)

from tqdm import tqdm
import numpy as np
from transformers import AlbertTokenizer, AlbertForMaskedLM, pipeline, FillMaskPipeline

from latency import utils

text = "Paris est la capitale de la [MASK]."


def main():
    unmasker = pipeline("fill-mask", model="qwant/fralbert-base")
    batch_length = 8
    prompt_batches = []
    for i in range(utils.random_folds):
        prompt_batch = []
        for j in range(batch_length):
            prompt = utils.prompts.get("8")[(i + j) % utils.random_folds].split()
            prompt[4] = "[MASK]"
            prompt = " ".join(prompt)
            prompt_batch.append(prompt)
        prompt_batches.append(prompt_batch)

    compute_latency = []
    for i in tqdm(range(utils.random_folds)):
        for prompt_batch in prompt_batches:
            start = time.time()
            _ = unmasker(prompt_batch)
            compute_latency.append(time.time() - start)

    print(
        f"Compute latency: {np.mean(compute_latency)/batch_length * 1000:.2f}ms - std {np.std(compute_latency)/batch_length * 1000:.2f}ms"
    )


if __name__ == "__main__":
    main()

import time
import random

random.seed(42)

from tqdm import tqdm
import numpy as np
from transformers import AlbertTokenizer, AlbertForMaskedLM, pipeline, FillMaskPipeline

from latency import utils

text = "Paris est la capitale de la [MASK]."


def main():
    # tokenizer = AlbertTokenizer.from_pretrained("qwant/fralbert-base")
    # model = AlbertForMaskedLM.from_pretrained(
    #    "qwant/fralbert-base",
    # )
    unmasker = pipeline("fill-mask", model="qwant/fralbert-base")

    compute_latency = []
    for i in tqdm(range(utils.random_folds * 10)):
        prompt = utils.prompts.get("8")[i % utils.random_folds].split()
        prompt[4] = "[MASK]"
        prompt = " ".join(prompt)
        start = time.time()
        _ = unmasker(prompt)
        compute_latency.append(time.time() - start)

    print(
        f"Compute latency: {np.mean(compute_latency) * 1000:.2f}ms - std {np.std(compute_latency) * 1000:.2f}ms"
    )


if __name__ == "__main__":
    main()

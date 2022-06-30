import time
import random

random.seed(42)

import torch
from transformers import AlbertTokenizer, AlbertForMaskedLM, pipeline, FillMaskPipeline

from latency import utils

text = "Paris est la capitale de la [MASK]."


def main():
    # tokenizer = AlbertTokenizer.from_pretrained("qwant/fralbert-base")
    # model = AlbertForMaskedLM.from_pretrained(
    #    "qwant/fralbert-base",
    # )
    unmasker = pipeline("fill-mask", model="qwant/fralbert-base")

    compute_latency = 0
    for _ in range(utils.random_folds):
        start = time.time()
        # use pipeline for equal benchmark with infinity
        # encoded_input = tokenizer(text, return_tensors="pt")
        # with torch.no_grad():
        #    raw_output = model(**encoded_input)
        # raw_output["input_ids"] = encoded_input["input_ids"]
        # decoded_output = unmasker.postprocess(raw_output)
        _ = unmasker(text)
        compute_latency += time.time() - start

    print(f"Compute latency: {compute_latency / utils.random_folds * 1000:.2f}ms")


if __name__ == "__main__":
    main()

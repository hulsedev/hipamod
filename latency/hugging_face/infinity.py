import requests
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from latency import utils

token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJpbmZpbml0eSIsImV4cCI6MTY1Njc3OTk5OCwiaWF0IjoxNjU2MTc1MTk4LCJpc3MiOiJodWdnaW5nZmFjZSIsImF1ZCI6Ikh1bHNlIn0.McrlpHUmiAldrcjtX7rw0GfTeVK0SKs0kxI64vO3LcWLp7qTDqtRAg8Ulv8jx3usMLijBw9N5FgpmRdEQXE5v3bcPYnRfvbDdqXQmawy6e60uQvLI90otzC95Yf3BNpa5R_frCPO4Vn2kqsL4ZFSZR5WsZ-Q69PfbloNx7rQ6taLTKUqLt6SI9GmbeBf7SvoVrQ_Ffs0SlP-aPiJgdSDhGV4JJiupZWpIYCQ3UUmjK2g_N_FHw3yeRRotXinGvGZnmzqbN5uTRlVe0ekItIm0aGUzfvcOYi7YHRtVl0rCDlUQ3HJ_eWvaf6JnlGtW_OAzuWDEfO0aTQgLieYvR1VYw"
url = "https://infinity.huggingface.co/{device}/{model}"
headers = {"Authorization": f"Bearer {token}"}
models = [
    ("ms-marco-minilm-l-6-v2", "cpu", "ranking"),
    ("ms-marco-minilm-l-6-v2", "gpu", "ranking"),
    ("all-minilm-l6-v2", "cpu", "embedding"),
    ("all-minilm-l6-v2", "gpu", "embedding"),
    (
        "distilbert-base-uncased-finetuned-sst-2-english",
        "cpu",
        "sequence-classification",
    ),
    (
        "distilbert-base-uncased-finetuned-sst-2-english",
        "gpu",
        "sequence-classification",
    ),
    ("bert-base-ner", "cpu", "token-classification"),
    ("bert-base-ner", "gpu", "token-classification"),
    ("bert-base-multilingual-uncased-sentiment", "cpu", "sequence-classification"),
    ("paraphrase-multilingual-minilm-l12-v2", "cpu", "embedding"),
]


def make_request(prompt, model, device, task):
    if task == "ranking":
        query = {"inputs": {"query": prompt, "documents": [prompt]}}
    else:
        query = {"inputs": prompt}

    tmp_url = url.format(device=device, model=model)
    tmp_start = time.time()
    resp = requests.post(
        tmp_url,
        json=query,
    )
    latency = time.time() - tmp_start
    assert resp.status_code == 200, f"Request failed {resp.status_code}"

    return resp.json(), latency


def main():
    results = {
        "model": [],
        "device": [],
        "task": [],
        "sequence_length": [],
        "latency": [],
        "network_latency": [],
        "compute_latency": [],
        "api": [],
    }
    for (model, device, task) in tqdm(models):
        for sequence_length in utils.prompts:
            latency, compute_latency = 0, 0
            for prompt in utils.prompts.get(sequence_length):
                if task == "ranking":
                    query = {"inputs": {"query": prompt, "documents": [prompt]}}
                else:
                    query = {"inputs": prompt}

                tmp_url = url.format(device=device, model=model)
                tmp_start = time.time()
                resp = requests.post(
                    tmp_url,
                    json=query,
                )
                latency += time.time() - tmp_start
                compute_latency += float(resp.headers["x-compute-time"])

            latency /= utils.random_folds
            compute_latency /= utils.random_folds

            results["model"].append(model)
            results["task"].append(task)
            results["device"].append(device)
            results["sequence_length"].append(sequence_length)
            results["latency"].append(latency)
            results["compute_latency"].append(compute_latency)
            results["network_latency"].append(latency - compute_latency)
            results["api"].append("Hugging Face Infinity")

    df = pd.DataFrame.from_dict(results)
    print(df.head())
    output_file = Path("latency/out/huggingface_infinity.csv")
    df.to_csv(output_file)


if __name__ == "__main__":
    main()

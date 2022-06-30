import requests
import time
from pathlib import Path
import json

import pandas as pd
from tqdm import tqdm

from latency import utils

token = "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"
url = "https://api-inference.huggingface.co/models/{model_path}"
headers = {"Authorization": f"Bearer {token}"}
# all models are not available on public inference endpoint
models = [
    # ("cross-encoder", "ms-marco-minilm-l-6-v2", "text-classification"),
    # ("sentence-transformers", "all-minilm-l6-v2", "sentence-similarity"),
    (
        None,
        "distilbert-base-uncased-finetuned-sst-2-english",
        "sequence-classification",
    ),
    # ("optimum", "bert-base-ner", "token-classification"),
    ("nlptown", "bert-base-multilingual-uncased-sentiment", "sequence-classification"),
    # ("sentence-transformers", "paraphrase-multilingual-minilm-l12-v2", "embedding"),
]


def main():
    results = {
        "model": [],
        "device": [],
        "task": [],
        "sequence_length": [],
        "latency": [],
        "api": [],
    }
    for (organization, model, task) in tqdm(models):
        for sequence_length in utils.prompts:
            latency = 0
            for idx, prompt in enumerate(utils.prompts.get(sequence_length)):
                if task == "ranking":
                    query = {"inputs": {"query": prompt, "documents": [prompt]}}
                else:
                    query = {"inputs": prompt}

                model_path = model if not organization else f"{organization}/{model}"
                tmp_url = url.format(model_path=model_path)
                tmp_start = time.time()
                resp = requests.post(tmp_url, headers=headers, data=json.dumps(query))
                latency += time.time() - tmp_start
                try:
                    assert (
                        resp.status_code == 200
                    ), f"Error querying model {resp.status_code}"
                except Exception as e:
                    break

            latency /= idx + 1

            results["model"].append(model)
            results["task"].append(task)
            results["device"].append("cpu")
            results["sequence_length"].append(sequence_length)
            results["latency"].append(latency)
            results["api"].append("Hugging Face Inference")

    df = pd.DataFrame.from_dict(results)
    print(df.head())
    output_file = Path("latency/out/huggingface_inference.csv")
    df.to_csv(output_file)


if __name__ == "__main__":
    main()

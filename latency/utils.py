import json
from pathlib import Path
import random

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

random.seed(42)

word_file = Path("latency/data/words.txt")
prompt_file = Path("latency/data/prompts.txt")
sequence_lengths = [8, 16, 32, 64, 128, 256, 512]
random_folds = 10
with open(word_file, "r") as f:
    words = set(map(lambda x: x.strip(), f.readlines()))

# generate prompts if not stored in file
if not prompt_file.is_file():
    print("No prompt file found, selecting some words...")
    prompts = {}
    for sequence_length in sequence_lengths:
        prompts[sequence_length] = []
        for _ in range(random_folds):
            prompts[sequence_length].append(
                " ".join(random.sample(words, sequence_length))
            )

    with open(prompt_file, "w") as f:
        json.dump(prompts, f)
else:
    with open(prompt_file, "r") as f:
        prompts = json.load(f)

models = [
    ("cross-encoder", "ms-marco-MiniLM-L-6-v2", "text-classification"),
    ("sentence-transformers", "all-MiniLM-L6-v2", "sentence-similarity"),
    (
        None,
        "distilbert-base-uncased-finetuned-sst-2-english",
        "sequence-classification",
    ),
    ("dslim", "bert-base-NER", "token-classification"),
    ("nlptown", "bert-base-multilingual-uncased-sentiment", "sequence-classification"),
    ("sentence-transformers", "paraphrase-multilingual-MiniLM-L12-v2", "embedding"),
]


def download_benchmark_models():
    base_dir = Path("latency/model/")
    if not base_dir.is_dir():
        base_dir.mkdir(parents=True, exist_ok=True)

    for org, model_name, task in models:
        model_ckpt = model_name if not org else f"{org}/{model_name}"
        model_path = base_dir.joinpath(model_name)
        if not model_path.is_dir():
            model_path.mkdir(parents=True, exist_ok=True)
        else:
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModel.from_pretrained(model_ckpt)

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    download_benchmark_models()

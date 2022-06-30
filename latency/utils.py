import json
from pathlib import Path
import random

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

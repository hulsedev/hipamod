from pathlib import Path

import evaluate
import datasets
import transformers
import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader

from latency.pilot.opt.evaluate import boolq


def main():
    """Evaluate performance of OPT model across 14 zero-shot tasks."""
    model_ckpt = "facebook/opt-350m"

    metrics = boolq.main(model_ckpt)


if __name__ == "__main__":
    main()

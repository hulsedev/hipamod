from pathlib import Path
import time
import random

random.seed(42)

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, pipeline
from optimum.onnxruntime.modeling_ort import ORTModelForQuestionAnswering

from latency import utils


def benchmark_deepsparse():
    # https://sparsezoo.neuralmagic.com/
    from sparsezoo.models import Zoo
    from deepsparse import compile_model

    batch_size = 16
    stub = "zoo:nlp/question_answering/obert-base/pytorch/huggingface/squad/base-none"

    # Download model and compile as optimized executable for your machine
    model = Zoo.download_model_from_stub(stub, override_parent_path="downloads")
    engine = compile_model(model, batch_size=batch_size)

    # Runs a benchmark
    inputs = model.data_inputs.sample_batch(batch_size=batch_size)
    benchmarks = engine.benchmark(inputs)
    print(benchmarks)


def get_batch(batch_size, length=128):
    return [" ".join(random.sample(utils.words, length)) for _ in range(batch_size)], [
        " ".join(random.sample(utils.words, int(length / 8))) for _ in range(batch_size)
    ]


def main():
    """Benchmark the performance of models taken from the Neural Magic
    Model Zoo."""

    base_dir = Path("latency/obert_surgeon/model/")
    model_paths = ["bert-base-uncased/"]
    model_filename = "model.onnx"

    hf_model_ckpt = "bert-base-uncased"
    onnx_model = ORTModelForQuestionAnswering.from_pretrained(
        base_dir.joinpath(hf_model_ckpt), file_name=model_filename
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_ckpt)
    classifier = pipeline("question-answering", model=onnx_model, tokenizer=tokenizer)

    compute_latency = []
    for i in tqdm(range(utils.random_folds)):
        context, question = get_batch(32)
        start = time.time()
        _ = classifier(question=question, context=context)
        compute_latency.append(time.time() - start)

    mean_throughput = np.mean(compute_latency) * 32
    std_throughput = np.std(compute_latency) * 32

    print(
        f"Compute latency: {mean_throughput:.2f} items/s - std {std_throughput:.2f} items/s"
    )


if __name__ == "__main__":
    main()

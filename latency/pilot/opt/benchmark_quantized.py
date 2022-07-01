import time
from pathlib import Path

from transformers import pipeline
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime.modeling_ort import ORTModelForCausalLM

from latency import utils
from latency.pilot.opt.compress.utils import OPTORTModelForCausalLM


def main(model_name):
    model_dir = Path(f"latency/pilot/opt/model/{model_name}/")
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    model_ckpt = f"facebook/{model_name}"
    onnx_model = OPTORTModelForCausalLM.from_pretrained(
        model_dir, file_name="model_quantized.onnx"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    classifier = pipeline("text-generation", model=onnx_model, tokenizer=tokenizer)

    compute_latency = []
    for i in tqdm(range(utils.random_folds)):
        start = time.time()
        _ = classifier(utils.prompts.get("8")[i % utils.random_folds])
        compute_latency.append(time.time() - start)

    print(
        f"Compute latency: {np.mean(compute_latency) * 1000:.2f}ms - std {np.std(compute_latency) * 1000:.2f}ms"
    )


if __name__ == "__main__":
    main("opt-125m")

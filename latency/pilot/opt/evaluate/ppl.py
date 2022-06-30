from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from pathlib import Path
import sys
import torch.nn.functional as F
from optimum.onnxruntime.modeling_ort import ORTModelForCausalLM
from transformers import pipeline, AutoModelForCausalLM
from onnxruntime import InferenceSession

from latency.pilot.opt.compress import export_onnx


def log_probs_with_ppl(model_ckpt, prompt, model_dir=None, model_filename=None):
    if model_dir:
        # base_model = AutoModelForCausalLM.from_pretrained(model_ckpt)
        # onnx_config = export_onnx.OPTOnnxConfig(base_model.config, task="causal-lm")

        session = InferenceSession(str(model_dir.joinpath(model_filename)))
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False)

        input_ids = tokenizer(prompt, return_tensors="np")
        outputs = session.run(None, dict(input_ids))

        logits = torch.from_numpy(outputs[0])
        print(logits.shape)

        arg_probs, _ = F.softmax(logits, dim=-1).max(-1)
        print("argmax probility:", arg_probs[0].cpu().detach().numpy())
        log_probs, tokens = F.log_softmax(logits, dim=-1).max(-1)
        print("argmax log probability:", log_probs[0].cpu().detach().numpy())
        sent = tokenizer.decode(
            tokens.squeeze().cpu().detach().numpy(),
            skip_special_tokens=False,
        )

        # TODO: add the loss to the outputs of the ONNX model
        print("argmax tokens:", sent)
        xentropy_loss = outputs[0]
        print("cross entropy loss:", xentropy_loss.item())
        ppl = torch.exp(xentropy_loss).item()
        print("ppl:", ppl)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_ckpt)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False)

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        logits = outputs.logits
        arg_probs, _ = F.softmax(logits, dim=-1).max(-1)
        print("argmax probility:", arg_probs[0].cpu().detach().numpy())
        log_probs, tokens = F.log_softmax(logits, dim=-1).max(-1)
        print("argmax log probability:", log_probs[0].cpu().detach().numpy())
        sent = tokenizer.decode(
            tokens.squeeze().cpu().detach().numpy(), skip_special_tokens=False
        )
        print("argmax tokens:", sent)
        xentropy_loss = outputs[0]
        print(dict(outputs).keys())
        print("cross entropy loss:", xentropy_loss.item())
        ppl = torch.exp(xentropy_loss).item()
        print("ppl:", ppl)
        assert False


if __name__ == "__main__":
    prompts = "There is a book on the desk."

    if len(sys.argv) > 1 and sys.argv[1] == "onnx":
        model_dir = Path("latency/pilot/opt/model/")
        model_id = "model.onnx"
        model_ckpt = "facebook/opt-350m"
        print(20 * "=" + str(model_id) + 20 * "=")
        log_probs_with_ppl(
            model_ckpt, prompts, model_dir=model_dir, model_filename=model_id
        )
    else:
        for model_id in [
            # "opt-125m",
            "opt-350m",
            "opt-1.3b",
            "opt-2.7b",
            # "opt-6.7b",
            # "opt-13b",
            # "opt-30b",
        ]:
            print(20 * "=" + model_id + 20 * "=")
            model_path = os.path.join("facebook", model_id)
            log_probs_with_ppl(model_path, prompts)

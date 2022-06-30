import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from onnxruntime import InferenceSession
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def log_probs_with_ppl(model_ckpt, prompt, model_dir=None, model_filename=None):
    if model_dir:
        session = InferenceSession(str(model_dir.joinpath(model_filename)))
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=False)
        config = AutoConfig.from_pretrained(model_ckpt)

        input_ids = tokenizer(prompt, return_tensors="np")
        outputs = session.run(None, dict(input_ids))
        logits = torch.from_numpy(outputs[0])

        arg_probs, _ = F.softmax(logits, dim=-1).max(-1)
        print("argmax probility:", arg_probs[0].cpu().detach().numpy())
        log_probs, tokens = F.log_softmax(logits, dim=-1).max(-1)
        print("argmax log probability:", log_probs[0].cpu().detach().numpy())
        sent = tokenizer.decode(
            tokens.squeeze().cpu().detach().numpy(),
            skip_special_tokens=False,
        )

        # extra step to compute the loss, since not included with onnx model
        labels = torch.from_numpy(input_ids.input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))

        print("argmax tokens:", sent)
        xentropy_loss = loss
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

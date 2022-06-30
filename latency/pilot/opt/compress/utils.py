from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from optimum.onnxruntime.modeling_ort import ORTModelForCausalLM


class OPTORTModelForCausalLM(ORTModelForCausalLM):
    def __init__(self, model=None, config=None, **kwargs):
        super().__init__(model, config, **kwargs)

        self.main_input_name = "input_ids"
        self.model_outputs = {
            output_key.name: idx
            for idx, output_key in enumerate(self.model.get_outputs())
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # converts pytorch inputs into numpy inputs for onnx
        onnx_inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        # run inference
        outputs = self.model.run(None, onnx_inputs)
        logits = torch.from_numpy(outputs[self.model_outputs["logits"]]).to(self.device)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        # converts output to namedtuple for pipelines post-processing
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)


class CustomOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)

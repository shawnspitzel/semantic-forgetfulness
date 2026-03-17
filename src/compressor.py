from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
from config import Config


class Compressor(nn.Module):
    """
    Frozen LLM backbone + LoRA + linear projection -> CE tensors.

    Same module handles both L1->L2 (input: token embeddings) and
    L2->L3 (input: L2 CE tensor) by varying target_c.
    EOS-token trick: append target_c EOS embeddings, collect hidden states at those positions.
    """

    def __init__(self, cfg: Config, peft_model=None, device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.adapter_name = "compressor"

        if peft_model is None:
            hf_cfg = AutoConfig.from_pretrained(cfg.model_name)
            self.hidden_dim: int = hf_cfg.hidden_size
            base = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            for p in base.parameters():
                p.requires_grad_(False)
            self.model = get_peft_model(base, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
            ), adapter_name=self.adapter_name)
        else:
            self.hidden_dim = peft_model.config.hidden_size
            self.model = peft_model

        self.projection = nn.Linear(self.hidden_dim, cfg.embed_dim, bias=False)
        self.to(self.device)
        # Match projection dtype to backbone (e.g. bfloat16 when loaded with torch_dtype=bfloat16)
        self.projection = self.projection.to(dtype=next(self.model.parameters()).dtype)

        # Extract after to(device) so _eos_embed lands on the correct device
        eos_id = self.model.config.eos_token_id or 0
        with torch.no_grad():
            self._eos_embed = (
                self.model.get_input_embeddings().weight[eos_id].detach().clone()
            )

    def compress(self, input_embeddings: torch.Tensor, target_c: int) -> torch.Tensor:
        """
        input_embeddings: [N, D] — token embeddings or CE tensor
        Returns: [target_c, embed_dim]
        """
        self.model.set_adapter(self.adapter_name)
        x = input_embeddings.to(self.device)
        N = x.shape[0]
        eos = self._eos_embed.unsqueeze(0).expand(target_c, -1)
        seq = torch.cat([x, eos], dim=0).unsqueeze(0)          # [1, N+C, D]
        out = self.model(inputs_embeds=seq, output_hidden_states=True)
        last_h = out.hidden_states[-1]                          # [1, N+C, hidden_dim]
        eos_h = last_h[0, N:, :]                                # [target_c, hidden_dim]
        return self.projection(eos_h)                           # [target_c, embed_dim]

    def parameters(self, recurse=True):
        adapter_params = [
            p for n, p in self.model.named_parameters()
            if f".{self.adapter_name}." in n and p.requires_grad
        ]
        return iter(adapter_params + list(self.projection.parameters()))

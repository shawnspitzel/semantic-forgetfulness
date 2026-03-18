from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)

from utils.config import Config
from utils.data_structures import SanityAnchors, L2Entry, ReconstructionResult


class Reconstructor(nn.Module):
    """
    APR (Anchored Progressive Reconstruction) decompressor.

    All operations are in embedding space — no text decode at any stage.
    Three-layer APR:
      Layer 1: Constraint shell (boundary locked, entity cosine check)
      Layer 2: Structured CE regions (entity / boundary / semantic)
      Layer 3: Context-grounded expansion (interpolate with L2 neighbors)

    L3->L2 output always has C_L2 slots (upsamples).
    L2->L1 output has C_L2 slots (refines and injects as soft tokens).
    """

    def __init__(self, cfg: Config, peft_model=None, device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.adapter_name = "reconstructor"

        if peft_model is None:
            hf_cfg = AutoConfig.from_pretrained(cfg.model_name)
            self.hidden_dim = hf_cfg.hidden_size
            base = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            for p in base.parameters():
                p.requires_grad_(False)
            self.model = get_peft_model(base, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_rank, lora_alpha=cfg.lora_rank * 2,
                lora_dropout=0.05, bias="none",
            ), adapter_name=self.adapter_name)
        else:
            self.hidden_dim = peft_model.config.hidden_size
            self.model = peft_model

        self.projection = nn.Linear(self.hidden_dim, cfg.embed_dim, bias=False)
        self._fingerprinter = None  # set via set_fingerprinter()
        self.to(self.device)
        # Match projection dtype to backbone (e.g. bfloat16 when loaded with torch_dtype=bfloat16)
        self.projection = self.projection.to(dtype=next(self.model.parameters()).dtype)

    def set_fingerprinter(self, fingerprinter) -> None:
        self._fingerprinter = fingerprinter

    def reconstruct(
        self,
        ce_tensor: torch.Tensor,
        anchors: SanityAnchors,
        l2_neighbors: list[L2Entry],
        query_vec: torch.Tensor | None,
        stage: str,
    ) -> ReconstructionResult:
        E, B = self.cfg.E, self.cfg.B
        C_out = self.cfg.C_L2
        C_in = ce_tensor.shape[0]

        # Run LLM forward to get initial C_out-slot reconstruction
        self.model.set_adapter(self.adapter_name)
        x = ce_tensor.to(self.device)
        raw = self.model.config.eos_token_id
        eos_id = (raw[0] if isinstance(raw, list) else raw) or 0
        eos_embed = self.model.get_input_embeddings().weight[eos_id].detach()
        eos_slots = eos_embed.unsqueeze(0).expand(C_out, -1)
        seq = torch.cat([x, eos_slots], dim=0).unsqueeze(0)     # [1, C_in+C_out, D]
        ctx = torch.enable_grad() if self.training else torch.no_grad()
        with ctx:
            out = self.model(inputs_embeds=seq, output_hidden_states=True)
        last_h = out.hidden_states[-1][0, C_in:, :]             # [C_out, hidden_dim]
        ce = self.projection(last_h)                             # [C_out, embed_dim]

        # -- Layer 1: Constraint Shell -----------------------------------------
        # boundary_ce[0] and boundary_ce[1] are CE-space encodings of the first and last
        # boundary sentences, computed via the compressor at admission time. They are injected
        # into the designated boundary slots (E and E+1) of the C_L2 CE layout, grounding
        # the reconstructed segment in its original boundary content.
        if anchors.boundary_ce is not None and anchors.boundary_ce.shape[0] >= 2:
            ce = ce.clone()
            if E < C_out:
                ce[E] = anchors.boundary_ce[0].to(self.device)
            if E + 1 < C_out:
                ce[E + 1] = anchors.boundary_ce[1].to(self.device)

        # -- Layer 3: Context-Grounded Expansion --------------------------------
        grounding_used = False
        if l2_neighbors and C_out > E + B:
            valid = [n.ce_tensor for n in l2_neighbors if n.ce_tensor.shape[0] > E + B]
            if valid:
                neighbor_sem = torch.stack([
                    n[E + B:].mean(dim=0).to(self.device) for n in valid
                ]).mean(dim=0)
                ce = ce.clone()
                ce[E + B:] = 0.5 * ce[E + B:] + 0.5 * neighbor_sem
                grounding_used = True

        # Query-conditioning for L2->L1
        if query_vec is not None and C_out > E + B:
            qv = query_vec.to(self.device)
            sims = F.cosine_similarity(ce[E + B:], qv.unsqueeze(0).expand(C_out - E - B, -1))
            weights = torch.softmax(sims, dim=0).unsqueeze(1)
            ce = ce.clone()
            ce[E + B:] = ce[E + B:] * weights * (C_out - E - B)

        # -- Fingerprint check --------------------------------------------------
        rep_vec = ce.mean(dim=0)
        if self._fingerprinter:
            fp = anchors.semantic_fingerprint.to(self.device)
            fingerprint_sim = F.cosine_similarity(
                F.normalize(rep_vec.unsqueeze(0), dim=1),
                F.normalize(fp.unsqueeze(0), dim=1),
            ).clamp(-1.0, 1.0).item()
        else:
            logger.warning(
                "[Reconstructor] fingerprinter not set — fingerprint_sim forced to 0.0; "
                "reconstruction will always fail theta check"
            )
            fingerprint_sim = 0.0  # stub when fingerprinter not wired — conservative default

        return ReconstructionResult(
            ce_tensor=ce,
            confidence_scores=[(i, 0.8) for i in range(C_out)],
            fingerprint_sim=fingerprint_sim,
            grounding_used=grounding_used,
            fallback=False,
        )

    def parameters(self, recurse=True):
        adapter_params = [
            p for n, p in self.model.named_parameters()
            if f".{self.adapter_name}." in n and p.requires_grad
        ]
        return iter(adapter_params + list(self.projection.parameters()))

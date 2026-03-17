from __future__ import annotations
import torch
from sentence_transformers import SentenceTransformer

class Fingerprinter:
    def __init__(self, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self._model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=str(self.device)
        )
        for p in self._model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        return self._model.encode(
            text, convert_to_tensor=True, normalize_embeddings=True
        ).to(self.device).float()

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        return self._model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        ).to(self.device).float()

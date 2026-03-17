from __future__ import annotations
import uuid, time
from typing import Optional
import torch
import torch.nn.functional as F

from sf.config import Config
from sf.cache_controller import CacheController
from sf.compressor import Compressor
from sf.reconstructor import Reconstructor
from sf.importance_scorer import ImportanceScorer
from sf.fingerprinter import Fingerprinter
from sf.entity_extractor import EntityExtractor
from sf.segmenter import Segmenter
from sf.data_structures import SanityAnchors


class InferenceLoop:
    def __init__(self, cfg: Config, device: str = "cpu", load_models: bool = False):
        self.cfg = cfg
        self.device = torch.device(device)
        self.session_id = str(uuid.uuid4())

        self.fingerprinter = Fingerprinter(device=device)
        self.entity_extractor = EntityExtractor(max_entities=cfg.E)
        self.segmenter = Segmenter(cfg)
        self.importance_scorer = ImportanceScorer(cfg)

        self.cache_controller = CacheController(
            cfg=cfg, session_id=self.session_id,
            embed_dim=cfg.embed_dim,
        )

        self.conversation_history: list[int] = []
        self._embedding_history: list[torch.Tensor] = []
        self._total_tokens_seen: int = 0

        self._llm = None
        self._tokenizer = None
        self._compressor: Optional[Compressor] = None
        self._reconstructor: Optional[Reconstructor] = None

        if load_models:
            self._load_models()

    def _load_models(self) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._llm = AutoModelForCausalLM.from_pretrained(self.cfg.model_name).to(self.device)
        for p in self._llm.parameters():
            p.requires_grad_(False)

        self._compressor = Compressor(self.cfg, device=self.device)
        self._reconstructor = Reconstructor(self.cfg, device=self.device)
        self._reconstructor.set_fingerprinter(self.fingerprinter)

        self.cache_controller.compress_fn = self._compressor.compress
        self.cache_controller.reconstruct_fn = self._reconstructor.reconstruct

    # ── Per-Turn API ─────────────────────────────────────────────────────

    def process_text(self, text: str) -> None:
        """Segment and admit text into the hierarchy."""
        if self._llm is None:
            self._process_text_mock(text)
        else:
            self._process_text_full(text)

    def _process_text_mock(self, text: str) -> None:
        """Fake token IDs for tests — no model calls."""
        words = text.split()
        token_ids = list(range(len(words)))
        for seg_ids in self.segmenter.segment(token_ids):
            sid = uuid.uuid4()
            n = len(seg_ids)
            embeddings = torch.randn(n, self.cfg.embed_dim)
            fp = self.fingerprinter.encode(text[:200])
            anchors = SanityAnchors(
                boundary_sentences=[text[:80], text[-80:]],
                entities=self.entity_extractor.extract(text[:200]),
                semantic_fingerprint=fp,
            )
            self.conversation_history.extend(seg_ids)
            self._embedding_history.append(embeddings.cpu())
            self._total_tokens_seen += n
            self.cache_controller.admit(
                sid, embeddings, 0.5, len(self.conversation_history) - n,
                fp, anchors, total_tokens_seen=self._total_tokens_seen,
            )

    def _process_text_full(self, text: str) -> None:
        tsp_idx = (self.cfg.tsp_layer_index if self.cfg.tsp_layer_index >= 0
                   else self._llm.config.num_hidden_layers // 2)
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        for seg_tokens in self.segmenter.segment(tokens):
            sid = uuid.uuid4()
            n = len(seg_tokens)

            # Current L1 ids for attention context
            l1_ids = []
            for entry in self.cache_controller.l1_entries():
                l1_ids.extend(entry.tokens.tolist())
            context_ids = l1_ids + seg_tokens
            input_ids = torch.tensor([context_ids], device=self.device)

            with torch.no_grad():
                out = self._llm(input_ids, output_attentions=True)

            attn = out.attentions[tsp_idx][0]
            importance = self.importance_scorer.score_from_attentions(
                attn, segment_start=len(l1_ids), segment_end=len(context_ids)
            )

            with torch.no_grad():
                emb_layer = self._llm.get_input_embeddings()
                embeddings = emb_layer(torch.tensor([seg_tokens], device=self.device))[0]

            seg_text = self._tokenizer.decode(seg_tokens)
            fp = self.fingerprinter.encode(seg_text)
            entities = self.entity_extractor.extract(seg_text)
            sents = [s.strip() for s in seg_text.split(".") if s.strip()]
            anchors = SanityAnchors(
                boundary_sentences=[sents[0] if sents else seg_text,
                                    sents[-1] if sents else seg_text],
                entities=entities,
                semantic_fingerprint=fp,
            )

            self.conversation_history.extend(seg_tokens)
            self._embedding_history.append(embeddings.cpu())
            self._total_tokens_seen += n
            self.cache_controller.admit(
                sid, embeddings, importance,
                len(self.conversation_history) - n,
                fp, anchors, total_tokens_seen=self._total_tokens_seen,
            )

    def generate_response(self, query: str, max_new_tokens: int = 200) -> str:
        if self._llm is None:
            self.process_text(query)
            return "[mock response — run with load_models=True for real inference]"

        # Admit query into cache hierarchy (consistent with mock path)
        self.process_text(query)

        query_fp = self.fingerprinter.encode(query)
        misses = self.cache_controller.detect_misses(query_fp)
        for miss in misses:
            meta = self.cache_controller.get_metadata(miss.segment_id)
            if meta and meta.fault_count >= self.cfg.leniency:
                self.cache_controller.promote_to_l1(miss.segment_id, query_fp)

        # KV rebuild: collect all L1 tokens in source_position order
        l1_ids: list[int] = []
        for entry in self.cache_controller.l1_entries():
            l1_ids.extend(entry.tokens.tolist())

        query_ids = self._tokenizer.encode(query, add_special_tokens=False)
        input_ids = torch.tensor([l1_ids + query_ids], device=self.device)

        with torch.no_grad():
            output = self._llm.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response_ids = output[0][input_ids.shape[1]:].tolist()
        response = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        self.process_text(response)
        return response

    @property
    def hit_rate_stats(self) -> dict:
        return self.cache_controller.hit_rate_stats

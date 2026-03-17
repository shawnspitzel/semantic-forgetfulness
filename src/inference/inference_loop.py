from __future__ import annotations
import uuid, time
from contextlib import nullcontext
from typing import Optional
import torch
import torch.nn.functional as F

from utils.config import Config
from memory.cache_controller import CacheController
from compression.compressor import Compressor
from compression.reconstructor import Reconstructor
from inference.importance_scorer import ImportanceScorer
from semantic.fingerprinter import Fingerprinter
from semantic.entity_extractor import EntityExtractor
from inference.segmenter import Segmenter
from utils.data_structures import SanityAnchors


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
        from peft import get_peft_model, LoraConfig, TaskType

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map=str(self.device),
        )
        for p in base.parameters():
            p.requires_grad_(False)

        if self.cfg.memory_enabled:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.cfg.lora_rank,
                lora_alpha=self.cfg.lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
            )
            peft_model = get_peft_model(base, lora_cfg, adapter_name="compressor")
            peft_model.add_adapter("reconstructor", lora_cfg)
            self._llm = peft_model
            self._compressor = Compressor(self.cfg, peft_model=peft_model, device=self.device)
            self._reconstructor = Reconstructor(self.cfg, peft_model=peft_model, device=self.device)
            self._reconstructor.set_fingerprinter(self.fingerprinter)
            self.cache_controller.compress_fn = self._compressor.compress
            self.cache_controller.reconstruct_fn = self._reconstructor.reconstruct
        else:
            self._llm = base

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
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if not self.cfg.memory_enabled:
            self.conversation_history.extend(tokens)
            self._total_tokens_seen += len(tokens)
            return

        tsp_idx = (self.cfg.tsp_layer_index if self.cfg.tsp_layer_index >= 0
                   else self._llm.config.num_hidden_layers // 2)

        for seg_tokens in self.segmenter.segment(tokens):
            sid = uuid.uuid4()
            n = len(seg_tokens)

            # Compute segment embeddings first (reused for both forward pass and admission)
            emb_layer = self._llm.get_input_embeddings()
            with torch.no_grad():
                embeddings = emb_layer(torch.tensor([seg_tokens], device=self.device))[0]

            # Build inputs_embeds from L1 context + current segment
            l1_entries = self.cache_controller.l1_entries()
            l1_embeds = [e.token_embeddings.to(self.device) for e in l1_entries]
            segment_start = sum(e.token_embeddings.shape[0] for e in l1_entries)
            segment_end = segment_start + n
            combined = torch.cat(l1_embeds + [embeddings], dim=0).unsqueeze(0)

            _ctx = self._llm.disable_adapter() if hasattr(self._llm, "disable_adapter") else nullcontext()
            with torch.no_grad(), _ctx:
                out = self._llm(inputs_embeds=combined, output_attentions=True)

            attn = out.attentions[tsp_idx][0]
            importance = self.importance_scorer.score_from_attentions(
                attn, segment_start=segment_start, segment_end=segment_end
            )

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

        self.process_text(query)

        # Format the current query as a proper instruction-tuned chat turn
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        formatted_ids = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        if self.cfg.memory_enabled:
            query_fp = self.fingerprinter.encode(query)
            misses = self.cache_controller.detect_misses(query_fp)
            for miss in misses:
                meta = self.cache_controller.get_metadata(miss.segment_id)
                if meta and meta.fault_count >= self.cfg.leniency:
                    self.cache_controller.promote_to_l1(miss.segment_id, query_fp)

            # Build inputs_embeds: L1 context embeddings + formatted query embeddings
            emb_layer = self._llm.get_input_embeddings()
            l1_embeds = [e.token_embeddings.to(self.device)
                         for e in self.cache_controller.l1_entries()]
            query_embeds = emb_layer(formatted_ids)[0]  # [T_fmt, hidden_dim]
            inputs_embeds = torch.cat(l1_embeds + [query_embeds], dim=0).unsqueeze(0)
            attention_mask = torch.ones(1, inputs_embeds.shape[1], device=self.device,
                                        dtype=torch.long)

            _ctx = self._llm.disable_adapter() if hasattr(self._llm, "disable_adapter") else nullcontext()
            with torch.no_grad(), _ctx:
                output = self._llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                )

            # generate() with inputs_embeds returns only new token IDs
            response_ids = output[0].tolist()
        else:
            # Flat context window: prepend history token IDs, then formatted query
            input_ids = formatted_ids
            if self.conversation_history:
                history_ids = torch.tensor([self.conversation_history], device=self.device)
                input_ids = torch.cat([history_ids, formatted_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)

            _ctx = self._llm.disable_adapter() if hasattr(self._llm, "disable_adapter") else nullcontext()
            with torch.no_grad(), _ctx:
                output = self._llm.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                )

            response_ids = output[0][input_ids.shape[1]:].tolist()

        response = self._tokenizer.decode(response_ids, skip_special_tokens=True)
        self.process_text(response)
        return response

    @property
    def hit_rate_stats(self) -> dict:
        return self.cache_controller.hit_rate_stats

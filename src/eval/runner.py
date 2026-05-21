from __future__ import annotations
import logging
import torch

logger = logging.getLogger(__name__)


class ModelRunner:
    """Loads the base LLM and runs inference with no memory framework."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_input_tokens: int = 32768,
        dry_run: bool = False,
    ) -> None:
        self.max_input_tokens = max_input_tokens
        self.dry_run = dry_run
        self._tokenizer = None
        self._model = None
        self._device_str = device

        if not dry_run:
            self._load(model_name, device)

    def _load(self, model_name: str, device: str) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info("Loading %s on %s …", model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        )
        self._model.eval()
        logger.info("Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 128) -> tuple[str, int]:
        """Returns (decoded_response, input_token_count)."""
        if self.dry_run:
            return "[DRY RUN]", 0

        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        n_input = input_ids.shape[1]

        if n_input > self.max_input_tokens:
            # Keep the tail so the question (always at end of prompt) is preserved.
            input_ids = input_ids[:, -self.max_input_tokens:]
            n_input = self.max_input_tokens

        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        response_ids = out[0][n_input:]
        response = self._tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response, n_input

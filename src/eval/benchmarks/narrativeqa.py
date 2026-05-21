"""NarrativeQA benchmark loader (deepmind/narrativeqa)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class NarrativeQASample:
    doc_id: str
    context: str
    question: str
    references: list[str]

    def format_prompt(self) -> str:
        return (
            "Read the following story and answer the question based only on the text provided.\n\n"
            f"Story:\n{self.context}\n\n"
            f"Question: {self.question}\n\n"
            "Answer:"
        )


def load_samples(max_samples: int | None = None) -> list[NarrativeQASample]:
    from datasets import load_dataset

    ds = load_dataset("deepmind/narrativeqa", split="test", trust_remote_code=True)
    samples: list[NarrativeQASample] = []
    for item in ds:
        if max_samples is not None and len(samples) >= max_samples:
            break
        samples.append(NarrativeQASample(
            doc_id=item["document"]["id"],
            context=item["document"]["text"],
            question=item["question"]["text"],
            references=[a["text"] for a in item["answers"]],
        ))
    return samples

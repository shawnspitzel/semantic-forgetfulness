"""LongBench benchmark loader (THUDM/LongBench).

Prompt templates and metric assignments follow the official LongBench evaluation
protocol from THUDM/LongBench on GitHub.
"""
from __future__ import annotations
from dataclasses import dataclass

# Metric to apply for each task.
TASK_METRIC: dict[str, str] = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "trec": "exact_match",
    "triviaqa": "f1",
    "samsum": "rouge_l",
    "passage_count": "exact_match",
    "passage_retrieval_en": "exact_match",
}

# Official LongBench prompt templates ({context} and {input} are substituted at runtime).
_PROMPTS: dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you "
        "can, using a single phrase or sentence if possible. If the question cannot be answered based "
        "on the information in the article, write \"unanswerable\". If the question is a yes/no "
        "question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single "
        "phrase or sentence if possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and "
        "do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news passages.\n\n"
        "News Passages:\n{context}\n\n"
        "Now, write a one-page summary of all the news passages.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output "
        "any other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in "
        "total?\n\n{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The answer is a number.\n\n"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Your task is to find the "
        "two paragraphs that are most semantically similar to the abstract.\n\n{context}\n\n"
        "The abstract is as follows:\n\n{input}\n\n"
        "Please enter the numbers of the two paragraphs (in the format "
        "\"The two most similar paragraphs are number X and number Y\") that are most semantically "
        "similar to the abstract."
    ),
}

DEFAULT_TASKS = ["narrativeqa", "qasper", "hotpotqa", "2wikimqa", "gov_report"]


@dataclass
class LongBenchSample:
    sample_id: str
    task: str
    context: str
    query: str
    references: list[str]

    def format_prompt(self) -> str:
        template = _PROMPTS.get(self.task, "{context}\n\nQuestion: {input}\nAnswer:")
        return template.format(context=self.context, input=self.query)


def load_samples(task: str, max_samples: int | None = None) -> list[LongBenchSample]:
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", name=task, split="test", trust_remote_code=True)
    samples: list[LongBenchSample] = []
    for item in ds:
        if max_samples is not None and len(samples) >= max_samples:
            break
        samples.append(LongBenchSample(
            sample_id=str(item["_id"]),
            task=task,
            context=item["context"],
            query=item["input"],
            references=list(item["answers"]),
        ))
    return samples

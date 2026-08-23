"""
PubMedQA: A Dataset for Biomedical Research Question Answering
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

TEMPLATE = r"""
Answer the following multiple choice question about medical knowledge given the context.
The entire content of your response should be of the following format: 'ANSWER: $LETTER'
(without quotes) where LETTER is one of {letters}.

{question}

{choices}
""".strip()

@task
def pubmedqa() -> Task:
    """Inspect Task implementation of the PubMedQA Eval"""
    dataset = hf_dataset(
        path="pubmed_qa",
        name="pqa_labeled",
        sample_fields=record_to_sample,
        split="train",
    )
    return Task(
        dataset=dataset,
        solver=[multiple_choice(template=TEMPLATE)],
        scorer=choice(),
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    choices = {"yes": "A", "no": "B", "maybe": "C"}
    context_list = record["context"]["contexts"]
    context_text = "\n".join(context_list)
    question = record["question"]
    return Sample(
        input=f"Context: {context_text}\nQuestion: {question}",
        target=choices[record["final_decision"].lower()],
        id=record["pubid"],
        choices=["yes", "no", "maybe"],
    )

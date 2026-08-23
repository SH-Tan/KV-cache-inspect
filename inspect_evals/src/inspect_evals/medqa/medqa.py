"""
MedQA: "What disease does this patient have? a large-scale open domain question answering dataset from medical exams"
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
def medqa() -> Task:
    """Inspect Task implementation of the MedQA Eval"""
    dataset = hf_dataset(
        path="openlifescienceai/medqa",
        sample_fields=record_to_sample,
        split="test",
    )
    return Task(
        dataset=dataset,
        solver=[multiple_choice(template=TEMPLATE)],
        scorer=choice(),
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    data = record["data"]
    options = data["Options"]
    question = data["Question"]
    correct_option = data["Correct Option"].strip()
    choice_letters = ["A", "B", "C", "D"]
    choices_text = [f"{letter}. {options[letter]}" for letter in choice_letters]
    return Sample(
        input=f"Question: {question}\n\nChoices:\n" + "\n".join(choices_text),
        target=correct_option,
        id=record["id"],
        choices=choice_letters,
        metadata={"subject_name": record["subject_name"]},
    )

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

@task
def tab_fact() -> Task:
    """Inspect Task implementation of the TabFact Eval"""
    dataset = hf_dataset(
        path="ibm-research/tab_fact",
        name="tab_fact",
        sample_fields=record_to_sample,
        trust=True,
        split="test",
    )
    return Task(dataset=dataset, solver=[multiple_choice()], scorer=choice())


def record_to_sample(record: dict[str, Any]) -> Sample:
    statement = record["statement"]
    table = record["table"]
    table_caption = table["caption"]
    table_text = "|" + "|".join(table["header"]) + "|\n" \
          + "|---" * len(table["header"]) + "|\n"
    for row in table["rows"]:
        table_text += "|" + "|".join(row) + "|\n"
    return Sample(
        input=f"Consider the following table:\n{table_text}\nTable Caption: {table_caption}\n\nIs the following statement entailed or refuted by the table?\n\nStatement: {statement}",
        target="A" if record["label"] == 1 else "B",
        choices=["True", "False"],
    )

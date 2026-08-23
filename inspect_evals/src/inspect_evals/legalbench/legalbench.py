from inspect_ai import Task, task
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from inspect_evals.legalbench.dataset import load_and_concatenate_datasets, resolve_subsets, Sample
from inspect_evals.legalbench.task_metadata import EXACT_MATCH_BALANCED_ACC_TASKS

@task
def legalbench(subsets: list[str] | str | None = None) -> Task:
    subsets = resolve_subsets(subsets or EXACT_MATCH_BALANCED_ACC_TASKS)
    for s in subsets:
        if s not in EXACT_MATCH_BALANCED_ACC_TASKS:
            raise ValueError(f"Subset {s} not in {EXACT_MATCH_BALANCED_ACC_TASKS}")
    combined_dataset = load_and_concatenate_datasets(subsets)
    return Task(
        dataset=combined_dataset,
        solver=[generate()],
        scorer=match(),
    )

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message

from inspect_evals.polyglot.data import aggregate_datasets
from inspect_evals.polyglot.scorer import decide_scorer


@task
def polyglot() -> Task:

    return Task(
        dataset=aggregate_datasets(),
        solver=generate(),
        scorer=[decide_scorer()],
        config=GenerateConfig(do_sample=False),
    )

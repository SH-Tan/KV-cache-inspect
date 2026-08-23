from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from inspect_evals.codeforces.data import aggregate_datasets
from inspect_evals.codeforces.scorer import decide_scorer


@task
def codeforces() -> Task:
    # Can be downloaded manually here: https://www.nltk.org/nltk_data/
    # Used for instruction following scoring on the livebench files below
    # https://github.com/LiveBench/LiveBench/blob/main/livebench/if_runner/instruction_following_eval/instructions_util.py
    # https://github.com/LiveBench/LiveBench/blob/main/livebench/if_runner/instruction_following_eval/instructions.py

    return Task(
        dataset=aggregate_datasets(),
        solver=generate(),
        scorer=[decide_scorer()],
        config=GenerateConfig(do_sample=False),
    )

"""
FinBen: An Holistic Financial Benchmark for Large Language Models

https://arxiv.org/pdf/2402.12659
https://github.com/The-FinAI/PIXIU?tab=readme-ov-file
"""
from inspect_ai import Task, task
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    choice,
    model_graded_fact,
    scorer,
    stderr,
)
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    multiple_choice,
    solver,
)
from inspect_ai.solver import generate as gen

from inspect_evals.finben.dataset import get_dataset

GRADER_MODEL = "openai/o4-mini"


@scorer(metrics=[accuracy(), stderr()])
def default_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        try:
            if len(state.choices) > 0:
                return await choice()(state, target)
            else:
                return await model_graded_fact(model=GRADER_MODEL)(state, target)
        except:
            return Score(value=0, answer="", metadata=state.metadata)
    return score

@solver
def default_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if len(state.choices) > 0:
            return await multiple_choice()(state, generate)
        else:
            return await gen(cache=True)(state, generate)
    return solve


@task
def finben(subsets: list[str] | str | None = None) -> Task:
    return Task(
        dataset=get_dataset(subsets),
        solver=[default_solver()],
        scorer=default_scorer(),
    )


@task
def finben_multiple_choice(subsets: list[str] | str | None = None) -> Task:
    return Task(
        dataset=get_dataset(subsets, "multiple_choice"),
        solver=[multiple_choice()],
        scorer=choice(),
    )


@task
def finben_qa(subsets: list[str] | str | None = None) -> Task:
    return Task(
        dataset=get_dataset(subsets, "qa"),
        solver=[gen(cache=True)],
        scorer=model_graded_fact(model=GRADER_MODEL),
    )


@task
def finben_flare_sm_acl() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-sm-acl"""
    return finben_multiple_choice("flare-sm-acl")


@task
def finben_flare_headlines() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-headlines"""
    return finben_multiple_choice("flare-headlines")


@task
def finben_fomc() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-fomc"""
    return finben_multiple_choice("flare-fomc")


@task
def finben_multifin_en() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-multifin-en"""
    return finben_multiple_choice("flare-multifin-en")


@task
def finben_ma() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-ma"""
    return finben_multiple_choice("flare-ma")


@task
def finben_finqa() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-finqa"""
    return finben_qa("flare-finqa")


@task
def finben_tatqa() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-tatqa"""
    return finben_qa("flare-tatqa")


@task
def finben_convfinqa() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-convfinqa"""
    return finben_qa("flare-convfinqa")


@task
def finben_german() -> Task:
    """https://huggingface.co/datasets/TheFinAI/flare-german"""
    return finben_multiple_choice("flare-german")


@task
def finben_taiwan() -> Task:
    """https://huggingface.co/datasets/TheFinAI/en-forecasting-taiwan"""
    return finben_multiple_choice("en-forecasting-taiwan")


@task
def finben_portoseguro() -> Task:
    """https://huggingface.co/datasets/TheFinAI/en-forecasting-portoseguro"""
    return finben_multiple_choice("en-forecasting-portoseguro")


@task
def finben_travelinsurance() -> Task:
    """https://huggingface.co/datasets/TheFinAI/en-forecasting-travelinsurance"""
    return finben_multiple_choice("en-forecasting-travelinsurance")

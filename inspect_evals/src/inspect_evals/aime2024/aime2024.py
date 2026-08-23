from typing import Any
import copy
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.scorer import Score, Scorer, match, Target, scorer, accuracy, stderr
from inspect_ai.solver import Solver, generate, prompt_template, TaskState

USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

_BOX_RE = re.compile(r"\\box(?:ed)?\s*\{\s*(.*?)\s*\}", re.DOTALL)

def extract_answer(text: str) -> str:
    m = _BOX_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    text = text.strip().rstrip(".")
    num_match = re.search(r"-?\d+(?:\.\d+)?", text)
    return num_match.group(0) if num_match else text

@scorer(metrics=[accuracy(), stderr()])
def or_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        assistant_msgs = [msg for msg in state.messages if msg.role == "assistant"]
        if not assistant_msgs:
            return Score(value=0, answer="", metadata=state.metadata)
        pred_raw = assistant_msgs[-1].text
        pred = extract_answer(pred_raw)
        is_correct = pred == target.text
        if is_correct:
            return Score(value=float(is_correct), answer=pred, metadata=state.metadata)
        else:
            result = await match()(state, target)
            return result
    return score

@task
def aime2024() -> Task:
    """Inspect Task implementation for the AIME 2024 benchmark."""
    base_ds = hf_dataset(
        path="Maxwell-Jia/AIME_2024",
        split="train",
        trust=True,
        sample_fields=record_to_sample,
    )
    dataset = boost_dataset(base_ds, factor=16)
    return Task(
        dataset=dataset,
        solver=aime2024_solver(),
        scorer=[or_scorer()],
    )

def boost_dataset(ds, factor: int = 16) -> MemoryDataset:
    """Return a dataset with each sample repeated `factor` times."""
    boosted = []
    for rep in range(factor):
        for s in ds:
            dup = copy.copy(s)
            dup.id = f"{s.id}_{rep}"
            boosted.append(dup)
    return MemoryDataset(samples=boosted, name="aime2024", location="aime2024")

def aime2024_solver() -> list[Solver]:
    return [prompt_template(USER_PROMPT_TEMPLATE), generate()]

def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["ID"],
        input=record["Problem"],
        target=str(record["Answer"]),
        metadata={"solution": record["Solution"]},
    )

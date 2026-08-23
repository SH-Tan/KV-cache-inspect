from typing import Any
import re
import copy

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.scorer import match, scorer, Score, Scorer, Target, accuracy, stderr
from inspect_ai.solver import generate, prompt_template, TaskState

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command. You should NOT include units in your answer, and answers should be integer.

Reasoning:
""".strip()

def strip_chinese(text: str) -> str:
    return ''.join(c for c in text if not ('\u4e00' <= c <= '\u9fff'))

_BOX_RE = re.compile(r"\\box(?:ed)?\s*\{\s*(.*?)\s*\}", re.DOTALL)

def extract_box(text: str) -> str:
    m = _BOX_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    text = text.strip().rstrip(".")
    num_match = re.search(r"-?\d+(?:\.\d+)?", text)
    return num_match.group(0) if num_match else text

@scorer(metrics=[accuracy(), stderr()])
def decide_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        assistant_messages = [msg for msg in state.messages if msg.role == "assistant"]
        if not assistant_messages:
            raise ValueError("No assistant message found in state.messages.")
        llm_answer = assistant_messages[-1].text
        llm_answer = re.sub(r"<think>.*?</think>", "", llm_answer, flags=re.DOTALL)
        clean_generated = strip_chinese(llm_answer)
        pred = extract_box(clean_generated)
        ref = target.text
        is_correct = pred == ref
        if is_correct:
            return Score(value=float(is_correct), answer=pred, metadata=state.metadata)
        state.output.completion = clean_generated
        result = await match(numeric=True)(state, target)
        return result
    return score

def boost_dataset(ds, factor: int = 16) -> MemoryDataset:
    """Return a dataset with each sample repeated `factor` times."""
    boosted = []
    for rep in range(factor):
        for s in ds:
            dup = copy.copy(s)
            dup.id = f"{s.id}_{rep}"
            boosted.append(dup)
    return MemoryDataset(samples=boosted, name="amc23", location="amc23")

@task
def amc23() -> Task:
    """Inspect Task definition for the AMC 2023 benchmark."""
    solver = [prompt_template(MATH_PROMPT_TEMPLATE), generate()]
    base_ds = hf_dataset(path="zwhe99/amc23", split="test", sample_fields=record_to_sample)
    dataset = boost_dataset(base_ds, factor=16)
    return Task(dataset=dataset, solver=solver, scorer=[decide_scorer()])


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(input=record["question"], target=str(int(record["answer"])), id=record["id"])

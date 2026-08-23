"""
Training Verifiers to Solve Math Word Problems

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
https://arxiv.org/abs/2110.14168
"""

from typing import Any
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match, scorer, Score, Scorer, Target, accuracy, stderr
from inspect_ai.solver import generate, prompt_template, system_message, TaskState

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

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

@task
def gsm8k(fewshot: int = 10, fewshot_seed: int = 42) -> Task:
    """Inspect Task definition for the GSM8K benchmark."""
    solver = [prompt_template(MATH_PROMPT_TEMPLATE), generate()]
    if fewshot:
        fewshots = hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            auto_id=True,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0,
            system_message(
                "\n\n".join([sample_to_fewshot(sample) for sample in fewshots])
            ),
        )
    return Task(
        dataset=hf_dataset(path="gsm8k", data_dir="main", split="test", sample_fields=record_to_sample),
        solver=solver,
        scorer=[decide_scorer()],
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})

def sample_to_fewshot(sample: Sample) -> str:
    if sample.metadata:
        return (
            f"{sample.input}\n\nReasoning:\n"
            + f"{sample.metadata['reasoning']}\n\n"
            + f"ANSWER: {sample.target}"
        )
    else:
        return ""

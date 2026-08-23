from typing import Any
import re

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, stderr
from inspect_ai.solver import Solver, generate, prompt_template, TaskState
from inspect_evals.math500.openmathinst_utils import process_results

USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

@task
def math500() -> Task:
    """Inspect Task implementation for the MATH-500 benchmark."""
    return Task(
        dataset=hf_dataset(path="zwhe99/MATH", split="math500", trust=True, sample_fields=record_to_sample),
        solver=math500_solver(),
        scorer=decide_scorer(),
    )

def math500_solver() -> list[Solver]:
    return [prompt_template(USER_PROMPT_TEMPLATE), generate()]

def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["id"],
        input=record["problem"],
        target=str(record["expected_answer"]),
        metadata={"solution": record["solution"]},
    )

@scorer(metrics=[accuracy(), stderr()])
def decide_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        assistant_messages = [msg for msg in state.messages if msg.role == "assistant"]
        if not assistant_messages:
            return Score(value=0, answer="", metadata=state.metadata)
        llm_answer = assistant_messages[-1].text
        llm_answer = re.sub(r"<think>.*?<\/think>", "", llm_answer, flags=re.DOTALL)
        value = process_results(
            llm_answer,
            target.text,
            response_extract_from_boxed=False,
            response_extract_regex=r"ANSWER: (.+)$",
        ) or process_results(
            llm_answer,
            target.text,
            response_extract_from_boxed=True,
        )
        return Score(value=value, answer=llm_answer, metadata=state.metadata)
    return score

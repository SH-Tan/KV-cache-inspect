import re
import json

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    scorer,
)
from inspect_ai.scorer import accuracy, stderr
from inspect_ai.solver import TaskState

from .code_reward import reward_fn_code

@scorer(metrics=[accuracy(), stderr()])
def decide_scorer() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        # Retrieve the last assistant message
        assistant_messages = [msg for msg in state.messages if msg.role == "assistant"]
        if not assistant_messages:
            raise ValueError("No assistant message found in state.messages.")
        llm_answer = assistant_messages[-1].text

        # Clean the LLM answer
        llm_answer = re.sub(r"<think>.*?<\/think>", "", llm_answer, flags=re.DOTALL)

        # Prepare the question object
        tests = state.metadata["tests"]

        # Compute the score
        value = reward_fn_code(llm_answer, tests)

        return Score(value=value, answer=llm_answer, metadata=state.metadata)

    return score

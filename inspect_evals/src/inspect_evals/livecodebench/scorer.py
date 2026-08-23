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
from .LiveCodeBench.lcb_runner.utils.extraction_utils import extract_code
from .LiveCodeBench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

def LCB_generation_process_results(question: dict, llm_answer: str, debug=False) -> int:

    extracted_answer = extract_code(model_output=llm_answer, lmstyle=None) # Missing out only on some slightly different handling for CodeLlamaInstruct from the original LiveCodeBench

    # code mostly from LiveCodeBench, with modifications.
    public_test_cases = json.loads(question['public_test_cases'])  # type: ignore

    try:
        private_test_cases = json.loads(question['private_test_cases'])  # type: ignore
    except:
        private_test_cases = json.loads(
            pickle.loads(
                zlib.decompress(
                    base64.b64decode(question['private_test_cases'].encode('utf-8'))  # type: ignore
                )
            )
        )  # type: ignore

    eval_sample = {
        "input_output": json.dumps(
            {
                "inputs": [
                    t.get("input", None)
                    for t in public_test_cases + private_test_cases
                ],
                "outputs": [
                    t.get("output", None)
                    for t in public_test_cases + private_test_cases
                ],
                "fn_name": question.get("func_name", None),
            }
        )
    }

    # print(eval_sample)
    # print(extracted_answer)

    metrics, results, metadata = codegen_metrics(
        [eval_sample],
        [[extracted_answer]],
        k_list=[1], # can't compute higher pass@ because we don't have more than one prediction.
        num_process_evaluate=1, # multiprocessing is handled at a higher level to parallelize multiple questions at once, so we don't want to complicate this with forking here.
        timeout=6, # default eval setting from livecodebench.
    )

    # print(metrics)
    # print(results)
    # print(metadata)

    if metrics['pass@1'] == 1.0:
        return 1
    else:
        return 0


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
        question_obj = state.metadata
        question_obj["question_id"] = state.sample_id

        # Compute the score
        value = LCB_generation_process_results(question_obj, llm_answer)

        return Score(value=value, answer=llm_answer, metadata=state.metadata)

    return score

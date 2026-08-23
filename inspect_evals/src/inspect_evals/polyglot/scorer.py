import re
import json
import os
import subprocess

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    scorer,
)
from inspect_ai.scorer import metric, accuracy, stderr, SampleScore
from inspect_ai.solver import TaskState
from pathlib import Path

@metric
def average_pass_ratio() -> float:
    def compute(scores) -> float:
        return sum(score.value['pass_ratio'] for score in scores) / len(scores)
    return compute

@metric
def fully_passed_rate() -> float:
    def compute(scores) -> float:
        return sum(score.value['fully_pass'] for score in scores) / len(scores)
    return compute

def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return ""
    return code_blocks[-1].strip()

def compute_pass_percentage_score(pytest_output: str) -> float:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned_output = ansi_escape.sub('', pytest_output)
    passed = cleaned_output.count("PASSED")
    failed = cleaned_output.count("FAILED")
    total = passed + failed
    if total == 0:
        return 0.0  # No test cases found

    return passed / total


def run_test(question: dict, code: str, debug=False, timeout=60) -> int:
    # Parse fields from the question dict
    problem_name = question["folder"]
    sol_file = question["sol_file"]
    test_file = question["test_file"]
    language = question["language"]

    problem_dir = f"./src/inspect_evals/polyglot/polyglot-benchmark/{language}/exercises/practice/{problem_name}"
    solution_path = os.path.join(problem_dir, sol_file)
    test_path = os.path.join(problem_dir, test_file)

    TEST_COMMANDS = {
        ".py": ["pytest", test_path, "--tb=short", "-q"],
        ".rs": ["cargo", "test", "--", "--include-ignored"],
        ".go": ["go", "test", "./..."],
        ".js": ["bash", "../../../../../npm-test.sh"],
        ".cpp": ["bash", "../../../../../cpp-test.sh"],
        ".java": ["./gradlew", "test"],
    }
    test_path_path = Path(test_path)
    extension = test_path_path.suffix

    # Ensure the directory exists
    os.makedirs(problem_dir, exist_ok=True)

    # Write solution code to file
    with open(solution_path, "w") as f:
        f.write(code)

    if debug:
        print(f"🔧 Wrote code to: {solution_path}")
        print(f"🧪 Running tests in: {test_path}")
    try:
        # Run pytest on the test file
        result = subprocess.run(
            TEST_COMMANDS[extension],
            cwd=problem_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        output = result.stdout

        score = compute_pass_percentage_score(output)

        if debug:
            print("✅ STDOUT:\n", output)
            print("⚠️ STDERR:\n", result.stderr)

        fully_pass = (1 if result.returncode == 0 else 0)
        return score, fully_pass

    except subprocess.TimeoutExpired:
        if debug:
            print("⏰ Timeout expired while running tests.")
        return 0.0, 0

    except Exception as e:
        if debug:
            print(f"❌ Unexpected error: {e}")
        return 0.0, 0


@scorer(metrics=[average_pass_ratio(), fully_passed_rate()])
def decide_scorer() -> Scorer:

    async def score(state: TaskState, target: Target) -> Score:
        # Retrieve the last assistant message
        assistant_messages = [msg for msg in state.messages if msg.role == "assistant"]
        if not assistant_messages:
            raise ValueError("No assistant message found in state.messages.")
        llm_answer = assistant_messages[-1].text

        # Clean the LLM answer
        llm_answer = re.sub(r"<think>.*?<\/think>", "", llm_answer, flags=re.DOTALL)

        code = extract_code_from_model(llm_answer)

        # Prepare the question object
        question_obj = state.metadata

        # Compute the score
        score, fully_pass = run_test(question_obj, code)

        return Score(value={"pass_ratio": score, "fully_pass": fully_pass}, answer=llm_answer, metadata=state.metadata)

    return score

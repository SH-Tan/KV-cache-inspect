
"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""
import json
import multiprocessing
import re
import time
from multiprocessing import Manager
from typing import List, Dict, Union
import random
import ast 
import os

from .run_test import run_test
from .reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType

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
        return None
    return code_blocks[-1].strip()


def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.

    Args:
        code (str): The input Python code.

    Returns:
        str: Cleaned code without the main execution block.
    """
    code_lines = code.split('\n')
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            # Check if we're out of the block (less indentation)
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def check_correctness(tests: Union[List[Dict[str, str]], Dict[str, List[str]]], code: str, timeout_per_test: int = 12, max_tests: int = 15) -> bool:
    """
    Check if generated code passes all test cases within a timeout period.

    Args:
        tests: Test cases in either list of dictionaries or dictionary of lists format
        code: Generated code to test
        test_fn: Function to run tests
        timeout: Maximum execution time in seconds before killing process

    Returns:
        bool: True if all tests pass, False otherwise

    Raises:
        AssertionError: If test results list is empty
    """
    if isinstance(tests, list):
        total_tests = len(tests)
        # print("******", total_tests)
        if total_tests > max_tests:
            # Sort indices by test input length and take the max_tests longest ones
            selected_indices = sorted(range(total_tests), key=lambda i: len(tests[i]['input']), reverse=True)[:max_tests]
            tests = [tests[i] for i in selected_indices]
        num_tests = len(tests)
    else:
        total_tests = len(tests['inputs'])
        if total_tests > max_tests:
            # Select the tests with the longest input length.
            selected_indices = sorted(range(total_tests), key=lambda i: len(tests['inputs'][i]), reverse=True)[:max_tests]
            # Create a new dict with only the selected test cases
            selected_tests = {
                'inputs': [tests['inputs'][i] for i in selected_indices],
                'outputs': [tests['outputs'][i] for i in selected_indices]
            }
            tests = selected_tests
        else:
            tests = {
                'inputs': [tests['inputs'][i] for i in range(total_tests)],
                'outputs': [tests['outputs'][i] for i in range(total_tests)]
            }
    
    try:
        results = run_test(tests,
                          test=code,
                          debug=False,
                          timeout=timeout_per_test)
    except Exception:
        return False
    # print(results)
    return all(r is True for r in results)

def safe_parse(x):
    """
    Safely parse an input into a Python object (dict, list, etc.).
    
    - If already a dict or list, return as-is.
    - If a string, try json.loads first.
    - If json.loads fails, fallback to ast.literal_eval.
    - Otherwise, raise an error.
    """
    if isinstance(x, (dict, list)):
        return x
    elif isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            x_fixed = re.sub(r"}\s*{", "}, {", x.strip())
            return ast.literal_eval(x_fixed)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")


class RewardCodeFn(RewardFn):
    """
    Reward function for evaluating code dataset answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the unit tests provided
    """
    def __call__(self, input: RewardInput) -> RewardOutput:

        model_response= input.model_response
        metadata = input.metadata

        tests = metadata
        if tests is None:
            print("No tests found in metadata")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        model_code = extract_code_from_model(model_response)
        if model_code is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        is_correct = False

        tests = safe_parse(tests)
        is_correct = check_correctness(tests, model_code)

        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def reward_fn_code(llm_solution: str, ground_truth: Dict, **kwargs):
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)
    reward_response = reward_fn(
        RewardInput(
            problem=None,
            problem_type=RewardType.CODE,
            data_source="",
            model_response=llm_solution,
            metadata=ground_truth
        ))
    if reward_response.is_correct:
        return 1.0
    else:
        return 0.0
  
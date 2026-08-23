from typing import Any, Dict, List
import json
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset


def aggregate_datasets() -> MemoryDataset:
    """Collects all Livebench samples into a single dataset."""
    all_samples: List[Sample] = []
    hf_ds = hf_dataset("livecodebench/code_generation", split="test", sample_fields=record_to_sample)
    all_samples = hf_ds.samples
    return MemoryDataset(
        samples=all_samples,
        name="livecodebench",
        location="livecodebench",
    )

def record_to_sample(record: Dict[str, Any]) -> Sample:
    return Sample(
        input=f"""Write Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end. \n {record["question_content"]}""",  # Single turn
        id=record["question_id"],
        metadata={
            "public_test_cases": record.get("public_test_cases"),
            "private_test_cases": record.get("private_test_cases"),
            "func_name": json.loads(record.get("metadata")).get("func_name", None)
        },
    )

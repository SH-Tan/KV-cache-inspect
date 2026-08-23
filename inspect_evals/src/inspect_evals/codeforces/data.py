from typing import Any, Dict, List
import json
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset


def aggregate_datasets() -> MemoryDataset:
    """Collects all Livebench samples into a single dataset."""
    seen_ids = set()
    all_samples: List[Sample] = []

    hf_ds = hf_dataset("chuxuan/RL-gen-codeforces-test", split="train", sample_fields=record_to_sample)

    for sample in hf_ds.samples:
        if sample.id in seen_ids:
            continue
        seen_ids.add(sample.id)
        all_samples.append(sample)

    return MemoryDataset(
        samples=all_samples,
        name="codeforces",
        location="codeforces",
    )


def record_to_sample(record: Dict[str, Any]) -> Sample:
    return Sample(
        input=f'''
        Title:\n {record["name"]} \n\n
        Problmen Description:\n {record["problem-description"]} \n\n
        Input Specification\n {record["input-specification"]} \n\n
        Output Specification\n {record["output-specification"]} \n\n
        Demo Input\n {record["demo-input"]} \n\n
        Demo Output\n {record["demo-output"]} \n\n
        Note\n {record["note"]} \n\n
        Write Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end.
        ''',  # Single turn
        id=f"{record['contestId']}-{record['index']}-{record['name']}",
        metadata={
            "tests": record.get("test_cases"),
        },
    )

from typing import Any, Dict, List
import json
import csv
from inspect_ai.dataset import MemoryDataset, Sample

def aggregate_datasets() -> MemoryDataset:
    """Collects samples from a local CSV file into a single MemoryDataset."""
    all_samples: List[Sample] = []
    csv_path = "./src/inspect_evals/polyglot/problem_data.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # if row.get("language", "").strip().lower() != "rust":
            #     continue
            sample = record_to_sample(row)
            all_samples.append(sample)

    return MemoryDataset(
        samples=all_samples,  # Only take first 5 if needed
        name="polyglot",
        location="polyglot",
    )



def record_to_sample(record: Dict[str, Any]) -> Sample:
    return Sample(
        input=record["instructions"],  # Single turn
        id=f"{record['folder']}-{record['language']}",
        metadata={
            "test_file": record["test_file"],
            "sol_file": record["sol_file"],
            "folder": record["folder"],
            "language": record["language"]
        },
    )

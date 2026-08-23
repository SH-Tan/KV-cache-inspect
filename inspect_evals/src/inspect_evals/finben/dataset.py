import concurrent.futures
from functools import partial, reduce
from typing import Any, Sequence

from inspect_ai.dataset import Dataset, Sample, hf_dataset

from inspect_evals.finben.task_metadata import (
    MULTIPLE_CHOICE_TASKS,
    QA_TASKS,
)


def get_dataset(
    subsets: set[str] | list[str] | str | None,
    scorer: str | None = None,
) -> Sequence[Sample]:
    if scorer == "multiple_choice":
        allowed_subsets = MULTIPLE_CHOICE_TASKS
    elif scorer == "qa":
        allowed_subsets = QA_TASKS
    else:
        allowed_subsets = set(MULTIPLE_CHOICE_TASKS) | set(QA_TASKS)
    subsets = resolve_subsets(subsets or allowed_subsets)
    for s in subsets:
        if s not in allowed_subsets:
            raise ValueError(f"Subset {s} not in {allowed_subsets}")
    return load_and_concatenate_datasets(subsets)


def load_dataset(
    subset: str,
    split: str = "test",
) -> Dataset:
    if subset in MULTIPLE_CHOICE_TASKS:
        record_to_sample = record_to_sample_multiple_choice
    elif subset in QA_TASKS:
        record_to_sample = record_to_sample_qa
    else:
        raise ValueError(f"Unknown subset: {subset}")

    return hf_dataset(
        "TheFinAI/" + subset,
        name="default",
        split=split,
        trust=True,
        sample_fields=partial(record_to_sample, subset=subset),
    )

def load_and_concatenate_datasets(
    subsets: list[str],
    split: str = "test",
) -> Sequence[Sample]:
    # loading datasets is very slow if not parallelized
    with concurrent.futures.ProcessPoolExecutor() as executor:
        datasets = list(
            executor.map(
                partial(
                    load_dataset,
                    split=split,
                ),
                subsets,
            )
        )
    return reduce(lambda a, b: list(a) + list(b), datasets)


def resolve_subsets(subsets: set[str] | list[str] | str) -> list[str]:
    if isinstance(subsets, str):
        return subsets.split(",")
    else:
        return list(subsets)


def get_question_from_record(record: dict[str, Any]) -> str:
    query = record["query"]
    assert isinstance(query, str), f"query is not a string: {query}"
    return query.removesuffix("\nAnswer:").strip()


def record_to_sample_multiple_choice(record: dict[str, Any], subset: str) -> Sample:
    correct_answer_idx = record["choices"].index(record["answer"])
    correct_answer_letter = chr(ord("A") + correct_answer_idx)
    sample = Sample(
        input=get_question_from_record(record),
        target=correct_answer_letter,
        choices=record["choices"],
        id=f"{subset}-{record['id']}",
        metadata={"original_ans": record["answer"]}
    )
    return sample


def record_to_sample_qa(record: dict[str, Any], subset: str) -> Sample:
    sample = Sample(
        input=get_question_from_record(record),
        target=record["answer"],
        id=f"{subset}-{record['id']}",
    )
    return sample

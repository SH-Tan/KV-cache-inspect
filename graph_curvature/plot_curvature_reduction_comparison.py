#!/usr/bin/env python3
"""Compare minimum and second-minimum curvature at layer and model ranking."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_evaluation_comparison import load_report


def plot_dataset(dataset):
    reports = {}
    for path in (dataset / "evaluation").glob("*/comparison.json"):
        if json.loads(path.read_text())["settings"]["score_source"] != "curvature":
            continue
        settings, cases = load_report(path, "curvature")
        ranking = settings.get("curvature_ranking", "layer")
        for method in ("curvature_min", "curvature_second_min"):
            if any(name == method for name, _ in cases):
                key = (method, ranking)
                if key in reports:
                    raise ValueError(f"Multiple reports for {key} in {dataset}")
                reports[key] = (settings, cases)

    methods = ("curvature_min", "curvature_second_min")
    rankings = ("layer", "model")
    for method in methods:
        for ranking in rankings:
            if (method, ranking) not in reports:
                raise ValueError(f"Missing {method} {ranking} report in {dataset}")
    reference = reports[("curvature_min", "layer")][0]
    baseline = reports[("curvature_min", "layer")][1][("baseline", 0.0)]
    ratios = None
    for method in methods:
        for ranking in rankings:
            settings, cases = reports[(method, ranking)]
            for field in ("task", "model", "prompt_count", "seed", "generation", "curvature_dir"):
                if settings[field] != reference[field]:
                    raise ValueError(f"Mismatched {field} for {method} {ranking}")
            if cases[("baseline", 0.0)] != baseline:
                raise ValueError(f"Mismatched baseline for {method} {ranking}")
            case_ratios = sorted(ratio for name, ratio in cases if name == method)
            if ratios is not None and case_ratios != ratios:
                raise ValueError(f"Mismatched ratios for {method} {ranking}")
            ratios = case_ratios

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for method, label, color in (("curvature_min", "Minimum", "#8c3b8c"),
                                 ("curvature_second_min", "Second minimum", "#d97706")):
        for ranking, style in (("layer", "o--"), ("model", "s-")):
            cases = reports[(method, ranking)][1]
            ax.plot(ratios, [cases[(method, ratio)] for ratio in ratios], style,
                    color=color, linewidth=2, markersize=6, label=f"{label} · {ranking}")
    ax.axhline(baseline, color="#6b7280", linestyle=":", linewidth=1.5,
               label=f"Uncompressed ({baseline:.2f}%)")
    task = reference["task"].removeprefix("longbench_").upper()
    ax.set(title=f"LongBench {task} · curvature reduction and ranking · {reference['prompt_count']} prompts",
           xlabel="KV removal ratio", ylabel="LongBench score (%)", xticks=ratios)
    ax.grid(alpha=0.2)
    ax.legend()
    output = dataset / "evaluation" / "comparison_min_second_min.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    for dataset in sorted(args.run_root.glob("longbench_*")):
        plot_dataset(dataset)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot each LCC method across last-row, attention-max, and attention-mean runs."""

import argparse
import json
from pathlib import Path

from plot_evaluation_comparison import load_report, plt


def load_layer_reports(run_root):
    """Merge split layer-level LCC reports using the existing report loader."""
    scores = {"native": {}, "curvature": {}}
    reference = None
    for path in sorted((run_root / "longbench_lcc/evaluation").glob("*/comparison.json")):
        settings = json.loads(path.read_text())["settings"]
        if settings.get("curvature_ranking", "layer") != "layer":
            continue
        source = settings["score_source"]
        settings, cases = load_report(path, source)
        if reference is None:
            reference = settings
        for key in ("task", "model", "prompt_count", "seed", "generation"):
            if settings[key] != reference[key]:
                raise ValueError(f"Evaluation settings differ: {key} in {path}")
        for key, value in cases.items():
            if key in scores[source] and scores[source][key] != value:
                raise ValueError(f"Conflicting scores for {key} in {path}")
            scores[source][key] = value
    return reference, scores


def plot_comparisons(runs, output_dir):
    """Save one figure per method, closing each before drawing the next."""
    reports = [(label, *load_layer_reports(root)) for label, root in runs]
    reference, reference_scores = reports[0][1:]
    baseline = reference_scores["curvature"][("baseline", 0.0)]
    press_methods = ("SnapKVPress", "ExpectedAttentionPress", "PyramidKVPress")
    for label, settings, scores in reports:
        for key in ("task", "model", "prompt_count", "seed", "generation"):
            if settings[key] != reference[key]:
                raise ValueError(f"Evaluation settings differ: {key} for {label}")
        if scores["curvature"][("baseline", 0.0)] != baseline:
            raise ValueError(f"Baseline differs for {label}")
        for method in press_methods:
            native = {ratio: value for (name, ratio), value in scores["native"].items()
                      if name == method}
            expected = {ratio: value for (name, ratio), value in reference_scores["native"].items()
                        if name == method}
            if native != expected:
                raise ValueError(f"Native {method} scores differ for {label}")

    output_dir.mkdir(parents=True, exist_ok=True)
    styles = (("#17649a", "o"), ("#dc6b21", "s"), ("#2a8c68", "^"))
    methods = ("curvature_min", "curvature_second_min", "curvature_mean", *press_methods)
    for method in methods:
        ratios = sorted(ratio for name, ratio in reference_scores["curvature"] if name == method)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for (label, _, scores), (color, marker) in zip(reports, styles):
            ax.plot(ratios, [scores["curvature"][(method, ratio)] for ratio in ratios],
                    color=color, marker=marker, linewidth=2, label=label)
        if method in press_methods:
            ax.plot(ratios, [reference_scores["native"][(method, ratio)] for ratio in ratios],
                    "x--", color="#444444", linewidth=1.5, label="Native attention")
        ax.axhline(baseline, color="#777777", linestyle=":", linewidth=1.5,
                   label=f"Uncompressed baseline ({baseline:.2f}%)")
        ax.set(title=f"LongBench LCC · {method.removesuffix('Press')} · "
                     f"{reference['prompt_count']} prompts",
               xlabel="KV removal ratio", ylabel="LCC code similarity score (%)", xticks=ratios)
        ax.grid(alpha=0.2)
        ax.legend()
        output = output_dir / f"comparison_{method.removesuffix('Press')}.png"
        fig.savefig(output, dpi=200)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last-row", type=Path, required=True)
    parser.add_argument("--attention-max", type=Path, required=True)
    parser.add_argument("--attention-mean", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_comparisons((("last row in A", args.last_row),
                      ("attention max", args.attention_max),
                      ("attention mean", args.attention_mean)), args.output)

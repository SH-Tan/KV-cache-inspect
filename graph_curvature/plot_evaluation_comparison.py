#!/usr/bin/env python3
"""Plot native attention and saved-curvature LongBench evaluation results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_report(path, expected_source):
    report = json.loads(path.read_text())
    if report["settings"]["score_source"] != expected_source:
        raise ValueError(f"Expected {expected_source} scores in {path}")
    cases = {(case["method"], case["removal_ratio"]): case["accuracy"] * 100
             for case in report["cases"]}
    return report["settings"], cases


def plot_curvature_ranking_comparison(layer_path, model_path, output_path):
    """Compare direct curvature ranking within layers and across the model."""
    layer_settings, layer = load_report(layer_path, "curvature")
    model_settings, model = load_report(model_path, "curvature")
    for key in ("task", "model", "prompt_count", "seed", "generation", "curvature_dir"):
        if layer_settings[key] != model_settings[key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    if (layer_settings.get("curvature_ranking", "layer") != "layer"
            or model_settings["curvature_ranking"] != "model"):
        raise ValueError("Expected layer-level and model-level curvature reports")
    baseline = layer[("baseline", 0.0)]
    if baseline != model[("baseline", 0.0)]:
        raise ValueError("Baseline scores differ between evaluations")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for method, color in (("curvature_min", "#8c3b8c"),
                          ("curvature_mean", "#2a8c68")):
        for cases, style, label in ((layer, "o--", "Layer-level (old)"),
                                     (model, "s-", "Model-level (new)")):
            ratios = sorted(ratio for name, ratio in cases if name == method)
            ax.plot(ratios, [cases[(method, ratio)] for ratio in ratios], style,
                    color=color, linewidth=2, label=f"{method} · {label}")
    ax.axhline(baseline, color="#777777", linestyle=":", linewidth=1.5,
               label=f"Uncompressed baseline ({baseline:.2f}%)")
    task_name = layer_settings["task"].removeprefix("longbench_").upper()
    ax.set(title=f"LongBench {task_name} · layer vs model curvature · "
                 f"{layer_settings['prompt_count']} prompts",
           xlabel="KV removal ratio", ylabel="LongBench score (%)",
           xticks=sorted({ratio for name, ratio in layer.keys() | model.keys()
                          if name in ("curvature_min", "curvature_mean")}))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_comparison(native_path, curvature_path, output_path):
    native_settings, native = load_report(native_path, "native")
    curvature_settings, curvature = load_report(curvature_path, "curvature")
    for key in ("task", "prompt_count", "seed"):
        if native_settings[key] != curvature_settings[key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    if native[("baseline", 0.0)] != curvature[("baseline", 0.0)]:
        raise ValueError("Baseline scores differ between evaluations")

    baseline = native[("baseline", 0.0)]
    task_name = native_settings["task"].removeprefix("longbench_").upper()
    score_label = {"longbench_lcc": "LCC code similarity score (%)",
                   "longbench_qasper": "Qasper F1 score (%)"}.get(
                       native_settings["task"], "LongBench score (%)")
    methods = ("SnapKVPress", "ExpectedAttentionPress", "PyramidKVPress")
    ratios = sorted({ratio for method, ratio in native if method == methods[0]})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, method in zip(axes.flat[:3], methods):
        ax.plot(ratios, [native[(method, ratio)] for ratio in ratios], "o-",
                color="#17649a", linewidth=2, label="Native attention")
        ax.plot(ratios, [curvature[(method, ratio)] for ratio in ratios], "s--",
                color="#dc6b21", linewidth=2, label="Saved curvature")
        ax.set_title(method.removesuffix("Press"))

    ax = axes.flat[3]
    for method, color, marker in (("curvature_min", "#8c3b8c", "D"),
                                  ("curvature_mean", "#2a8c68", "^")):
        ax.plot(ratios, [curvature[(method, ratio)] for ratio in ratios], marker + "-",
                color=color, linewidth=2, label=method)
    ax.set_title("Direct curvature ranking")

    for ax in axes.flat:
        ax.axhline(baseline, color="#777777", linestyle=":", linewidth=1.5,
                   label=f"Uncompressed baseline ({baseline:.2f}%)")
        ax.set_xticks(ratios)
        ax.grid(alpha=0.2)
    for ax in axes.flat:
        ax.legend(fontsize=9, loc="upper left" if ax is axes.flat[1] else "lower left")
    fig.suptitle(f"LongBench {task_name} · {native_settings['prompt_count']} prompts", fontsize=16)
    fig.supxlabel("KV removal ratio")
    fig.supylabel(score_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = {"SnapKVPress": "#17649a", "ExpectedAttentionPress": "#dc6b21",
              "PyramidKVPress": "#2a8c68"}
    for method in methods:
        name = method.removesuffix("Press")
        ax.plot(ratios, [native[(method, ratio)] for ratio in ratios], "o-",
                color=colors[method], linewidth=2, label=f"{name} · native attention")
        ax.plot(ratios, [curvature[(method, ratio)] for ratio in ratios], "s--",
                color=colors[method], linewidth=2, label=f"{name} · saved curvature")
    for method, color, marker in (("curvature_min", "#8c3b8c", "D"),
                                  ("curvature_mean", "#795329", "^")):
        ax.plot(ratios, [curvature[(method, ratio)] for ratio in ratios], marker + "-",
                color=color, linewidth=2, label=method)
    ax.axhline(baseline, color="#777777", linestyle=":", linewidth=1.5,
               label=f"Uncompressed baseline ({baseline:.2f}%)")
    ax.set(title=f"LongBench {task_name} · all methods · {native_settings['prompt_count']} prompts",
           xlabel="KV removal ratio", ylabel=score_label, xticks=ratios)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.savefig(output_path.with_name("comparison_all_methods.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path, help="Native-score comparison.json")
    parser.add_argument("curvature", type=Path, help="Curvature-score comparison.json")
    parser.add_argument("output", type=Path, help="Panel PNG path; also writes comparison_all_methods.png")
    parser.add_argument("--model-curvature", type=Path,
                        help="Model-level comparison.json; also writes comparison_curvature_layer_vs_model.png")
    args = parser.parse_args()
    plot_comparison(args.native, args.curvature, args.output)
    if args.model_curvature is not None:
        plot_curvature_ranking_comparison(
            args.curvature, args.model_curvature,
            args.output.with_name("comparison_curvature_layer_vs_model.png"))

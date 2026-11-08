#!/usr/bin/env python3
"""Compare saved last-query attention and curvature for every key token."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import torch


SCORE_NAMES = ("Mean attention", "Minimum curvature", "Mean curvature")


def reduce_token_scores(record):
    """Reduce heads, preserving every token and ignoring undefined curvatures."""
    attention = record["attention_scores"].float().numpy().mean(axis=0, dtype=np.float64)
    curvature = record["curvature_scores"]["A"].numpy()
    finite = np.isfinite(curvature)
    count = finite.sum(axis=0)
    minimum = np.min(curvature, axis=0, where=finite, initial=np.inf)
    minimum[count == 0] = np.nan
    average = np.full(attention.shape, np.nan)
    np.divide(np.sum(curvature, axis=0, where=finite, dtype=np.float64), count,
              out=average, where=count > 0)
    return (attention, minimum, average), count


def percentile_ranks(values):
    """Map ascending ranks to [0, 1], assigning tied values their mean rank."""
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    count = int(finite.sum())
    if count:
        result[finite] = (rankdata(values[finite]) - 1) / (count - 1) if count > 1 else 0.5
    return result


def score_correlations(attention, curvature):
    """Keep raw correlations and importance agreement distinct."""
    valid = np.isfinite(attention) & np.isfinite(curvature)
    a, c = attention[valid], curvature[valid]
    pearson = spearman = float("nan")
    if a.size > 1 and np.ptp(a) > 0 and np.ptp(c) > 0:
        pearson = float(np.corrcoef(a, c)[0, 1])
        spearman = float(np.corrcoef(rankdata(a), rankdata(c))[0, 1])
    return {"finite_tokens": int(valid.sum()), "pearson": pearson, "spearman": spearman,
            "importance_spearman": -spearman}


def correlation_label(statistics):
    return (f"Raw curvature: Pearson r = {statistics['pearson']:+.3f}, "
            f"Spearman ρ = {statistics['spearman']:+.3f}\n"
            f"Importance (−curvature): Spearman ρ = {statistics['importance_spearman']:+.3f}")


def plot_layer(record, scores, statistics, output):
    """Use two large scatter panels to compare every token's importance."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for index, (ax, stats, color) in enumerate(zip(axes, statistics, ("#d97706", "#15803d"))):
        valid = np.isfinite(scores[0]) & np.isfinite(scores[index + 1])
        # Rank on the same finite tokens used for the correlation.
        x = percentile_ranks(scores[0][valid])
        y = percentile_ranks(-scores[index + 1][valid])
        ax.scatter(x, y, s=5, alpha=0.22, color=color, edgecolors="none", rasterized=True)
        ax.plot([0, 1], [0, 1], color="#374151", linestyle="--", linewidth=1,
                label="Identical importance ranks")
        ax.set_title(f"{SCORE_NAMES[index + 1]}\n{correlation_label(stats)}", fontsize=12, pad=12)
        ax.set(xlabel="Attention importance percentile",
               ylabel="Curvature importance percentile", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right", fontsize=10)
    figure.suptitle(
        f"Layer {record['layer_id']:03d} | {scores[0].size:,} tokens | attention vs curvature\n"
        "Higher attention / more negative curvature = more important. "
        "Percentile 1 = most important; 0 = least important.", fontsize=14)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_token_profiles(record, scores, statistics, output):
    """Keep raw signed scores in token order and save a single PNG."""
    figure, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True, layout="constrained")
    token_indices = np.arange(scores[0].size)
    for index, (ax, values, color) in enumerate(zip(axes, scores, ("#2563eb", "#d97706", "#15803d"))):
        ax.plot(token_indices, values, linewidth=0.45, color=color)
        ax.set_ylabel(SCORE_NAMES[index])
        ax.set_xlim(0, scores[0].size - 1)
        ax.grid(alpha=0.2)
        if index == 0:
            ax.set_yscale("log")
            ax.set_title("Mean attention across heads — higher is more important", loc="left")
        else:
            ax.set_yscale("symlog", linthresh=0.1)
            ax.invert_yaxis()
            stats = statistics[index - 1]
            ax.set_title(
                f"{SCORE_NAMES[index]} across heads — more negative is more important (upward)\n"
                f"Raw Pearson r = {stats['pearson']:+.3f} | raw Spearman ρ = {stats['spearman']:+.3f} | "
                f"importance Spearman ρ = {stats['importance_spearman']:+.3f}",
                loc="left", fontsize=11)
    axes[-1].set_xlabel("Key token index")
    figure.suptitle(
        f"Layer {record['layer_id']:03d} | prompt {record['prompt_id']:05d} | "
        f"all {scores[0].size:,} tokens | final query index {record['query_index']}\n"
        "Raw scores in original token order; curvature axes are reversed so importance increases upward.",
        fontsize=14)
    with matplotlib.rc_context({"path.simplify": False}):
        figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_agreement(records, output, prompt_id):
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    for ax, metric in zip(axes, ("importance_spearman", "pearson")):
        for reduction, label in (("min", "Minimum curvature"), ("mean", "Mean curvature")):
            values = [r[f"{reduction}_{metric}"] for r in records]
            ax.plot([r["layer_id"] for r in records], values,
                    marker="o", markersize=3, label=f"{label} (mean {np.nanmean(values):+.3f})")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set(xlabel="Layer index", ylabel="Correlation", ylim=(-1.05, 1.05))
        ax.set_title("Importance Spearman: attention vs −curvature" if metric == "importance_spearman"
                     else "Raw Pearson: attention vs curvature")
        ax.grid(alpha=0.2)
        ax.legend()
    figure.suptitle(f"Prompt {prompt_id:05d}: attention and curvature correlation by layer\n"
                   "Agreement means positive importance correlation and negative raw correlation.",
                   fontsize=13)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/analysis/token_importance")
    parser.add_argument("--prompt-id", type=int, default=0)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "analysis" / "token_importance"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.run_dir.glob(f"layer_*/prompt_{args.prompt_id:05d}.pt"))
    if not paths:
        parser.error("No matching per-layer prompt files found")
    torch.set_num_threads(1)
    summaries = []
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        scores, _ = reduce_token_scores(record)
        statistics = [score_correlations(scores[0], s) for s in scores[1:]]
        stem = f"layer_{record['layer_id']:03d}_prompt_{args.prompt_id:05d}"
        plot_layer(record, scores, statistics, output / f"{stem}.png")
        plot_token_profiles(record, scores, statistics, output / f"{stem}_tokens.png")
        summary = {"layer_id": record["layer_id"], "prompt_id": record["prompt_id"],
                   "token_count": record["token_count"], "query_index": record["query_index"]}
        for reduction, stats in zip(("min", "mean"), statistics):
            summary.update({f"{reduction}_{key}": value for key, value in stats.items()})
        summaries.append(summary)
        print(f"Layer {record['layer_id']:03d}: importance Spearman "
              f"min={statistics[0]['importance_spearman']:+.4f}, "
              f"mean={statistics[1]['importance_spearman']:+.4f}", flush=True)
        del record, scores
    plot_agreement(summaries, output / f"prompt_{args.prompt_id:05d}_agreement.png", args.prompt_id)
    (output / "README.md").write_text(
        "# Per-token attention and curvature comparison\n\n"
        "Each layer has a two-panel importance scatter plot and three raw token-score profiles. "
        "All reductions are across heads for each key token, using only the saved final query. "
        "The token profiles preserve every position without sampling or binning. "
        "Each figure is saved only as PNG.\n\n"
        "Importance percentiles run from 0 (least important) to 1 (most important). Higher "
        "attention and lower (more negative) curvature mean higher importance. Ties receive "
        "their average rank. In scatter plots, points on the diagonal have identical importance "
        "percentiles; points away from it indicate disagreement. Each scatter includes all "
        "tokens finite in both measures.\n\n"
        "Every figure annotates attention vs raw curvature Pearson and Spearman correlations, "
        "plus attention vs negative curvature Spearman importance agreement. Negative raw "
        "correlation and positive importance correlation both indicate agreement. Token profiles "
        "keep original signed values; curvature axes are reversed so more important tokens "
        "are higher. Attention uses a log axis; curvature uses symlog with threshold 0.1.\n\n"
        "Undefined curvature heads are excluded from the minimum and mean. Tokens with no "
        "finite curvature remain undefined. No CSV files are written.\n\n"
        "Only one layer file is loaded at a time, on CPU; no model or GPU is needed.\n\n"
        "Recreate with:\n\n```bash\n"
        f"MPLCONFIGDIR=/tmp/kv-cache-matplotlib /home/tans5/anaconda3/envs/prune_llm/bin/python "
        f"graph_curvature/plot_token_importance.py {args.run_dir}\n```\n"
    )
    with (output / "README.md").open("a") as stream:
        stream.write(f"\n[Correlation overview](prompt_{args.prompt_id:05d}_agreement.png)\n\n"
                     "| Layer | Correlation scatter | Token scores | Importance ρ: min | Importance ρ: mean |\n"
                     "| --- | --- | --- | --- | --- |\n")
        for summary in summaries:
            layer = summary["layer_id"]
            stem = f"layer_{layer:03d}_prompt_{args.prompt_id:05d}"
            stream.write(f"| {layer} | [PNG]({stem}.png) | [PNG]({stem}_tokens.png) | "
                         f"{summary['min_importance_spearman']:+.4f} | "
                         f"{summary['mean_importance_spearman']:+.4f} |\n")
    print(f"Saved {len(summaries)} layer comparisons and agreement summary to {output}")


if __name__ == "__main__":
    main()

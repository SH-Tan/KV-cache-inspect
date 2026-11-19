#!/usr/bin/env python3
"""Plot original curvature and attention together for sampled heads and tokens."""

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import torch


def sample_token_indices(curvature, rng, count=100, near_zero_count=50, fraction=0.1):
    """Sample both negative-minimum tails, including the most-negative token."""
    minimum = np.min(curvature, axis=0, where=np.isfinite(curvature), initial=np.inf)
    negative = np.flatnonzero(np.isfinite(minimum) & (minimum < 0))
    other_count = count - near_zero_count
    tail_size = int(np.ceil(negative.size * fraction))
    near_size, other_size = max(near_zero_count, tail_size), max(other_count, tail_size)
    if near_size + other_size > negative.size:
        raise ValueError("Not enough negative-minimum tokens for two disjoint sampling pools")
    ordered = negative[np.argsort(minimum[negative], kind="stable")]
    near_zero_pool, other_pool = ordered[-near_size:], ordered[:other_size]
    near_zero = np.sort(rng.choice(near_zero_pool, size=near_zero_count, replace=False))
    # Include the extreme minimum, then sample the rest from the most-negative tail.
    other = np.sort(np.concatenate((other_pool[:1], rng.choice(
        other_pool[1:], size=other_count - 1, replace=False))))
    indices = np.concatenate((near_zero, other))
    ranges = [(float(minimum[group].min()), float(minimum[group].max()))
              for group in (near_zero, other)]
    return indices, ranges


def sample_context_tokens(curvature, rng, fraction=0.1):
    """Reserve both endpoints, then sample disjoint curvature and interior groups."""
    token_count = curvature.shape[1]
    sampled, ranges = sample_token_indices(curvature[:, 6:-6], rng, 60, 30, fraction)
    sampled += 6
    remaining = np.setdiff1d(np.arange(6, token_count - 6), sampled)
    middle = np.sort(rng.choice(remaining, size=10, replace=False))
    indices = np.concatenate((np.arange(6), sampled[:30], middle,
                              sampled[30:], np.arange(token_count - 6, token_count)))
    return indices, ranges


def select_disagreeing_tokens(curvature, attention, count=50):
    """Select interior tokens with the largest attention/curvature importance rank gap."""
    minimum = np.min(curvature, axis=0, where=np.isfinite(curvature), initial=np.inf)
    mean_attention = attention.mean(axis=0, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(minimum) & np.isfinite(mean_attention))
    attention_rank = rankdata(-mean_attention[valid], method="average")
    curvature_rank = rankdata(minimum[valid], method="average")
    rank_gap = np.abs(attention_rank - curvature_rank)
    interior = np.flatnonzero((valid > 0) & (valid < curvature.shape[1] - 1))
    if interior.size < count:
        raise ValueError(f"Need {count} interior tokens with finite scores, found {interior.size}")
    selected = interior[np.argsort(-rank_gap[interior], kind="stable")[:count]]
    indices = np.concatenate(([0], np.sort(valid[selected]), [curvature.shape[1] - 1]))
    percentile_gap = 100 * rank_gap[selected] / max(1, valid.size - 1)
    groups = [(1, "Token\n0", 24),
              (count, f"{count} largest attention / curvature importance rank gaps\n"
               f"Absolute percentile-rank gap: {percentile_gap.min():.1f}–{percentile_gap.max():.1f} points", 36),
              (1, "Last\ntoken", 24)]
    return indices, groups


def select_joint_extreme_tokens(curvature, attention, per_group=25, cutoff=0.75):
    """Select disjoint high/high and low/low tokens using matched heads."""
    valid = np.isfinite(curvature) & np.isfinite(attention)
    counts = valid.sum(axis=1, keepdims=True)
    attention_percentile, curvature_percentile = (
        (rankdata(np.where(valid, values, np.nan), axis=1, nan_policy="omit") - 0.5) / counts
        for values in (attention, curvature)
    )
    high = np.where(valid, np.minimum(attention_percentile, curvature_percentile), -np.inf)
    low = np.where(valid & (curvature < 0),
                   np.minimum(1 - attention_percentile, 1 - curvature_percentile), -np.inf)
    scores = np.stack((high.max(axis=0), low.max(axis=0)))
    heads = np.stack((high.argmax(axis=0), low.argmax(axis=0)))
    scores[:, [0, -1]] = -np.inf
    selected = [[], []]
    used = set()
    # Assign strongest cases first; each token belongs to at most one group.
    for flat_index in np.argsort(-scores.ravel(), kind="stable"):
        group, token = divmod(int(flat_index), curvature.shape[1])
        if scores[group, token] < cutoff:
            break
        if len(selected[group]) < per_group and token not in used:
            selected[group].append(token)
            used.add(token)
        if all(len(tokens) == per_group for tokens in selected):
            break
    high_tokens, low_tokens = (np.sort(np.asarray(tokens, dtype=np.int64)) for tokens in selected)
    indices = np.concatenate(([0], high_tokens, low_tokens, [curvature.shape[1] - 1]))
    selected_heads = np.concatenate(([-1], heads[0, high_tokens], heads[1, low_tokens], [-1]))
    groups = [(1, "Token\n0", 24)]
    if high_tokens.size:
        high_cutoff = 100 * scores[0, high_tokens].min()
        groups.append((high_tokens.size, f"{high_tokens.size} high attention + high curvature\n"
                       f"Both ≥ {high_cutoff:.1f}th percentile\n(in the circled head)", 30))
    if low_tokens.size:
        low_cutoff = 100 * (1 - scores[1, low_tokens].min())
        groups.append((low_tokens.size, f"{low_tokens.size} low attention + low curvature\n"
                       f"Both ≤ {low_cutoff:.1f}th percentile\n(curvature < 0 in circled head)", 30))
    groups.append((1, "Last\ntoken", 24))
    return indices, groups, selected_heads


def token_groups(token_count, near_zero_count, ranges, context=False):
    """Build shared panel labels once for either token-selection layout."""
    counts = (30, 30) if context else (near_zero_count, token_count - near_zero_count)
    separator, fontsize = ("\n", 30) if context else (" ", 36)
    groups = [
        (count, f"{count} {label} tokens\nmin curvature{separator}[{low:.3g}, {high:.3g}]", fontsize)
        for count, label, (low, high) in zip(counts, ("near-zero", "strongly negative"), ranges)
    ]
    if context:
        groups = [(6, "First 6\n(0–5)", 24), groups[0],
                  (10, "10 random\ninterior tokens", 24), groups[1],
                  (6, "Last 6\n(last−5\nto last)", 24)]
    return groups


def token_group_layout(groups, token_count, highlight_heads=False):
    """Compute shared dividers and labels once for both heatmap panels."""
    dividers = np.cumsum([count for count, _, _ in groups])[:-1] - 0.5
    labels = []
    start = 0
    for count, label, fontsize in groups:
        center = (start + count / 2) / token_count
        start += count
        if not highlight_heads:
            labels.append((center, label, fontsize, None))
        elif "attention" in label:
            label_x = 0.25 if "high attention" in label else 0.75
            labels.append((label_x, label, fontsize, center))
    if highlight_heads and not any("low attention" in label for _, label, _ in groups):
        labels.append((0.75, "0 low attention + low curvature\nNo qualifying bottom-quartile tokens", 30, None))
    return dividers, labels


def plot_head_heatmaps(curvature, attention, token_indices, groups, title, output, selected_heads=None):
    """Draw aligned raw-score panels with shared token columns and readable labels."""
    dividers, labels = token_group_layout(groups, len(token_indices), selected_heads is not None)
    marked_columns = np.flatnonzero(selected_heads >= 0) if selected_heads is not None else None
    figure, axes = plt.subplots(2, 1, figsize=(32, 28), sharex=True)
    for ax, score_name, values in zip(axes, ("Curvature", "Attention"), (curvature, attention)):
        finite = values[np.isfinite(values)]
        if score_name == "Curvature":
            low, high = float(finite.min()), float(finite.max())
            norm = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high) if low < 0 < high else Normalize(low, high)
            cmap = plt.get_cmap("RdBu_r").copy()
        else:
            norm = LogNorm(vmin=float(finite[finite > 0].min()), vmax=float(finite.max()))
            cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#bdbdbd")
        mesh = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest",
                         cmap=cmap, norm=norm)
        if marked_columns is not None:
            for size, color in ((170, "black"), (110, "white")):
                ax.scatter(marked_columns, selected_heads[marked_columns], s=size, facecolors="none",
                           edgecolors=color, linewidths=2, zorder=3)
        ax.set_yticks(np.arange(values.shape[0]), np.arange(values.shape[0]), fontsize=18)
        ax.set_ylabel("Head index", fontsize=40, labelpad=18)
        ax.set_title(f"Original {score_name.lower()}", fontsize=44,
                     pad=150 if len(groups) > 2 else 122, loc="left")
        for divider in dividers:
            ax.axvline(divider, color="#666666", linewidth=2)
        for label_x, label, fontsize, center in labels:
            ax.text(label_x, 1.025, label, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=fontsize, linespacing=1.25)
            if center is not None:
                ax.annotate("", xy=(center, 1), xytext=(label_x, 1.02), xycoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "color": "#666666"})
        scale = ("logarithmic color scale" if isinstance(norm, LogNorm) else
                 "linear on each side of zero" if isinstance(norm, TwoSlopeNorm) else "linear scale")
        colorbar = figure.colorbar(mesh, ax=ax, pad=0.02, fraction=0.03)
        colorbar.set_label(f"Original {score_name.lower()} value\n({scale})", fontsize=36, labelpad=24)
        ticks = norm.inverse(np.linspace(0, 1, 7))
        colorbar.set_ticks(ticks, labels=[f"{value:.0e}" if isinstance(norm, LogNorm) else f"{value:.3g}"
                                        for value in ticks])
        colorbar.minorticks_off()
        colorbar.ax.tick_params(labelsize=32, pad=10)
    axes[-1].set_xticks(np.arange(len(token_indices)), token_indices, rotation=90, fontsize=17)
    axes[-1].set_xlabel("Original key-token index — identical columns in both panels", fontsize=40, labelpad=24)
    figure.suptitle(title, fontsize=40, y=0.985, linespacing=1.3)
    if selected_heads is not None:
        figure.text(0.5, 0.025, "Circled cells: selected head; percentiles compare tokens within each head",
                    ha="center", fontsize=26)
    figure.subplots_adjust(left=0.065, right=0.865, bottom=0.10, top=0.82, hspace=0.40)
    temporary = output.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=110)
    plt.close(figure)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--near-zero-count", type=int, default=50)
    parser.add_argument("--fraction", type=float, default=0.1,
                        help="Fraction of negative minima used for each sampling tail")
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--context-tokens", action="store_true",
                        help="New 82-token view: first/last 6, 30 near-zero, 30 strongly negative, 10 random interior")
    mode.add_argument("--rank-disagreement", action="store_true",
                      help="50 largest mean-attention vs minimum-curvature rank gaps, plus first/last token")
    mode.add_argument("--joint-extremes", action="store_true",
                      help="Up to 25 high/high + 25 low/low same-head quartile tokens, plus first/last token")
    parser.add_argument("--resume", action="store_true", help="Skip existing combined PNGs; use the same settings")
    args = parser.parse_args()
    if not 0 < args.fraction < 0.5 or not 0 < args.near_zero_count < args.count:
        parser.error("fraction must be in (0, 0.5) and near-zero-count must be between 0 and count")
    torch.set_num_threads(1)
    output = args.output_dir or args.run_dir / (
        "joint_extreme_heatmaps" if args.joint_extremes else
        "rank_disagreement_heatmaps" if args.rank_disagreement else
        "context_head_heatmaps" if args.context_tokens else "sampled_head_heatmaps")
    output.mkdir(parents=True, exist_ok=True)
    if args.joint_extremes:
        sampling = (
            "Each panel has up to 52 distinct tokens: token 0, up to 25 high-attention/high-curvature "
            "tokens, up to 25 low-attention/low-curvature tokens, and the last token. For each head separately, "
            "rank attention and raw signed curvature across tokens using (average rank − 0.5) / "
            "finite pair count. High/high score = min(attention percentile, curvature percentile). "
            "Low/low score = min(1 − attention percentile, 1 − curvature percentile), requiring "
            "negative curvature. Each token uses its strongest qualifying head. Greedily assign "
            "cases by decreasing score, enforcing at most 25 tokens per group and no duplicate tokens; "
            "ties prefer high/high, then earlier token position. Both percentiles must be "
            "in the top quartile for high/high, or the bottom quartile for low/low. "
            "Never fill shortages with nonqualifying tokens; actual group sizes are shown. "
            "Sort positions within each group. The circled cell identifies the selected head "
            "in both panels; labels give the weakest selected joint percentile in each group. "
            "This selects directional contradictions within heads, rather than mixing a mean "
            "attention score and a minimum curvature from potentially different heads.")
    elif args.rank_disagreement:
        sampling = (
            "Each panel has 52 distinct tokens: token 0, the 50 interior tokens with the largest "
            "absolute importance-rank gaps, and the last token. Rank all tokens with finite scores "
            "by mean attention across heads (descending) and minimum finite curvature across heads "
            "(ascending, equivalent to descending negative minimum curvature). Use average ranks "
            "for ties. Select the largest absolute rank differences, breaking ties by earlier token "
            "position. Display the selected 50 in original token order. Labels show the absolute "
            "rank gap divided by (finite token count − 1), in percentage points.")
    elif args.context_tokens:
        sampling = (
            f"Seed: {args.seed}. Each panel has 82 distinct tokens. Groups, left to right: "
            "first 6 (indices 0–5), 30 near-zero negative minima, 10 random interior tokens, "
            "30 strongly negative minima, last 6 (last−5 through last, inclusive). "
            "Reserve the first/last 6 before sampling. Sample the 10 random interior tokens "
            "excluding all prior selections.")
    else:
        sampling = (
            f"Seed: {args.seed}. First {args.near_zero_count} columns: near-zero negative minima. "
            f"Remaining {args.count - args.near_zero_count} columns: strongly negative minima.")
    lines = ["# Combined head × token heatmaps", "",
             "Prompts and layers reuse each dataset's analysis/token_importance/sampled_README.md.",
             sampling,
             "" if args.rank_disagreement or args.joint_extremes else
             f"Randomly sample near-zero tokens from the closest-to-zero {args.fraction:.0%} of "
             f"eligible negative minima, and strongly negative tokens from the most-negative "
             f"{args.fraction:.0%}, always including the most-negative eligible token. "
             "Each tail is rounded up and contains at least as many candidates as sampled tokens. "
             "Token minima are computed across finite head curvatures. Sort positions within each group. "
             "Dividers separate groups; both panels share all columns. "
             "Group labels show the sampled minimum-curvature ranges.",
             "Each PNG contains original curvature above original attention, both head × token. "
             "The saved final query, original head order, and token columns match across panels. "
             "No head averaging or score normalization. Attention uses a logarithmic color scale "
             "to distinguish small positive scores; colorbar ticks show original probabilities. "
             "Curvature uses separate linear intervals below/above zero when both signs occur. "
             "Colorbars use raw values shared across all heads. Gray means undefined curvature.",
             "PNG figures only, with this index; no JSON or NPZ files. "
             "CPU only; one memory-mapped source record and one figure at a time.", "",
             "| Dataset | Prompt | Layer | Combined curvature and attention |",
             "| --- | --- | --- | --- |"]
    figures = 0
    for dataset_index, dataset in enumerate(sorted(args.run_dir.glob("longbench_*"))):
        selection = (dataset / "analysis/token_importance/sampled_README.md").read_text().splitlines()
        prompts = ast.literal_eval(next(line for line in selection if line.startswith("Random seed:")).split("prompts: ")[1])
        layers = ast.literal_eval(next(line for line in selection if line.startswith("Layers:")).split(": ", 1)[1])
        for prompt_id in prompts:
            folder = output / dataset.name / f"prompt_{prompt_id:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            for layer_id in layers:
                stem = folder / f"layer_{layer_id:03d}"
                figure_path = Path(f"{stem}_heatmaps.png")
                lines.append(f"| {dataset.name} | {prompt_id} | {layer_id} | "
                             f"[PNG]({figure_path.relative_to(output).as_posix()}) |")
                figures += 1
                if not (args.resume and figure_path.exists()):
                    source = dataset / f"layer_{layer_id:03d}" / f"prompt_{prompt_id:05d}.pt"
                    record = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
                    curvature_scores = record["curvature_scores"]["A"].numpy()
                    selected_heads = None
                    if args.joint_extremes or args.rank_disagreement:
                        attention_scores = record["attention_scores"].float().numpy()
                        if args.joint_extremes:
                            indices, groups, selected_heads = select_joint_extreme_tokens(
                                curvature_scores, attention_scores)
                        else:
                            indices, groups = select_disagreeing_tokens(curvature_scores, attention_scores)
                        attention = attention_scores[:, indices]
                        del attention_scores
                    else:
                        rng = np.random.default_rng([args.seed, dataset_index, prompt_id, layer_id])
                        if args.context_tokens:
                            indices, ranges = sample_context_tokens(curvature_scores, rng, args.fraction)
                        else:
                            indices, ranges = sample_token_indices(
                                curvature_scores, rng, args.count, args.near_zero_count, args.fraction)
                        groups = token_groups(len(indices), args.near_zero_count, ranges, args.context_tokens)
                        attention = record["attention_scores"][:, indices].float().numpy()
                    curvature = curvature_scores[:, indices]
                    title = (f"{dataset.name}  |  prompt {prompt_id:05d}  |  layer {layer_id:03d}  |  "
                             f"final query {record['query_index']}\n"
                             f"{curvature.shape[0]} heads × {len(indices)} tokens per panel")
                    if args.joint_extremes:
                        title += "  |  up to 25 high/high + 25 low/low"
                    del record, curvature_scores
                    plot_head_heatmaps(curvature, attention, indices, groups, title, figure_path,
                                       selected_heads=selected_heads)
            print(f"{dataset.name} prompt {prompt_id:05d}: {len(layers)} combined figures", flush=True)
            (output / "README.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {figures} combined figures to {output}", flush=True)


if __name__ == "__main__":
    main()

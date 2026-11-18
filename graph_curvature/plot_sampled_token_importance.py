#!/usr/bin/env python3
"""Draw token profiles for five random prompts and evenly spaced layers per dataset."""

import argparse
from pathlib import Path
import random

import torch

from plot_token_importance import plot_token_profiles, reduce_token_scores, score_correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    rng = random.Random(args.seed)
    for dataset in sorted(args.run_dir.glob("longbench_*")):
        layer_dirs = sorted(dataset.glob("layer_[0-9][0-9][0-9]"))
        layer_ids = (*range(0, 4), *range(10, 15), *range(20, 25),
                     *range(31, 35), *range(38, 40))
        selected_layers = [layer_dirs[i] for i in layer_ids]
        prompt_ids = sorted(int(path.stem.split("_")[1])
                            for path in layer_dirs[0].glob("prompt_*.pt"))
        selected_prompts = sorted(rng.sample(prompt_ids, 5))
        index = [f"# {dataset.name} token profiles", "",
                 f"Random seed: {args.seed}; prompts: {selected_prompts}",
                 f"Layers: {[int(layer.name.split('_')[1]) for layer in selected_layers]}", ""]
        for prompt_id in selected_prompts:
            output = dataset / "analysis" / "token_importance" / f"prompt_{prompt_id:05d}"
            output.mkdir(parents=True, exist_ok=True)
            index.append(f"## Prompt {prompt_id:05d}")
            index.append("")
            for layer in selected_layers:
                name = f"{layer.name}_prompt_{prompt_id:05d}_tokens.png"
                index.append(f"- [Layer {layer.name[-3:]}](prompt_{prompt_id:05d}/{name})")
                if (output / name).is_file() and not args.overwrite:
                    continue
                record = torch.load(layer / f"prompt_{prompt_id:05d}.pt",
                                    map_location="cpu", weights_only=True, mmap=True)
                scores, _ = reduce_token_scores(record)
                statistics = [score_correlations(scores[0], values) for values in scores[1:]]
                plot_token_profiles(record, scores, statistics, output / name)
                print(dataset.name, f"prompt {prompt_id:05d}", layer.name, flush=True)
                del record, scores
            index.append("")
        (dataset / "analysis" / "token_importance" / "sampled_README.md").write_text(
            "\n".join(index))


if __name__ == "__main__":
    main()

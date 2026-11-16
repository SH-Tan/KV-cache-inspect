#!/usr/bin/env python3
"""Compare model-wide curvature retention across five prompts per benchmark."""

from pathlib import Path
import argparse
import json
import random
import re
import textwrap

import numpy as np
import torch
from transformers import AutoTokenizer

BENCHMARKS = ('trec', 'qasper', 'lcc', 'multi_news')


def minimum_curvature(record):
    values = record['curvature_scores']['A'].numpy()
    finite = np.isfinite(values)
    minimum = np.min(values, axis=0, where=finite, initial=np.inf)
    minimum[~finite.any(axis=0)] = np.nan
    return minimum


def top_mask(scores, fraction):
    """Select exactly the highest-scoring fraction of token positions."""
    count = max(1, int(scores.size * fraction))
    mask = np.zeros(scores.size, dtype=bool)
    mask[np.argsort(-scores, kind='stable')[:count]] = True
    return mask


def top_differences(curvature, attention, curvature_mask, attention_mask):
    """Top quarter of exclusive selections by absolute importance-rank gap."""
    ranks = []
    for scores in (curvature, attention):
        rank = np.empty(scores.size, dtype=np.int64)
        rank[np.argsort(-scores, kind='stable')] = np.arange(scores.size)
        ranks.append(rank)
    different = np.flatnonzero(curvature_mask ^ attention_mask)
    gap = np.abs(ranks[0][different] - ranks[1][different])
    count = int(np.ceil(different.size * 0.25))
    return different[np.argsort(-gap, kind='stable')[:count]]


def original_words(ids, positions, tokenizer):
    """Map saved token positions to complete words/punctuation in decoded text."""
    text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == ids.tolist(), 'Decoded prompt changed token positions'
    spans = list(re.finditer(r"\w+(?:['’-]\w+)*|[^\w\s]", text))
    starts = np.array([span.start() for span in spans])
    ends = np.array([span.end() for span in spans])
    words = {}
    for position in positions:
        start, end = encoded['offset_mapping'][position]
        first = np.searchsorted(ends, start, side='right')
        last = np.searchsorted(starts, end, side='left')
        words[int(position)] = (text[starts[first]:ends[last - 1]]
                                if first < last else '<whitespace>')
    return words


def analyze_prompt(path):
    record = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    scores = -minimum_curvature(record)
    scores[~np.isfinite(scores)] = -np.inf
    result = None
    if record['layer_id'] == 0:
        attention = record['attention_scores'].float().numpy().mean(axis=0, dtype=np.float64)
        curvature50, attention50 = top_mask(scores, 0.5), top_mask(attention, 0.5)
        result = {'curvature50': curvature50, 'attention50': attention50,
                  'differences25': top_differences(scores, attention, curvature50, attention50)}
    return scores, result


def layer_retention(scores):
    """Percent retained per layer under the evaluator's 50% model-wide budget."""
    layers, length = scores.shape
    count = layers * max(1, length // 2)
    selected = np.argsort(-scores.ravel(), kind='stable')[:count]
    return 100 * np.bincount(selected // length, minlength=layers) / length


def write_layer_report(path, title, labels, retention):
    lines = [title, '50% model-wide compression; rank all layer/token pairs together.',
             'Importance = -minimum curvature across heads, using the final prompt query.',
             'Values = % of each layer retained. Higher = more tokens ranked important.',
             'Ties prefer earlier layers, then earlier tokens. Undefined scores rank last.',
             'Means weight prompts/datasets equally. Layer numbers start at 0.', '',
             'Five layers retaining the most tokens:']
    mean = retention.mean(axis=0)
    for label, row in zip([*labels, 'Mean'], [*retention, mean]):
        top = np.argsort(-row, kind='stable')[:5]
        lines.append(f'{label}: ' + ', '.join(f'L{layer} ({row[layer]:.1f}%)' for layer in top))
    lines.extend(['', 'Layer | ' + ' | '.join(f'{label:>10}' for label in [*labels, 'Mean'])])
    for layer in range(retention.shape[1]):
        values = [*retention[:, layer], mean[layer]]
        lines.append(f'{layer:5d} | ' + ' | '.join(f'{value:9.1f}%' for value in values))
    path.write_text('\n'.join(lines) + '\n')


def write_differences(path, name, prompt_ids, ids_list, results, tokenizer):
    lines = [f'{name.upper()}: LAYER 0, 50% COMPRESSION',
             'Five prompts (seed 29). Lowest min curvature vs highest mean attention; final query.',
             'Keep floor(tokens/2) each. Show top 25% of disagreeing positions by absolute rank gap (rounded up).',
             'Ties prefer earlier positions. Words [zero-based token positions], grouped in prompt order.',
             'Subwords expand to original words; repeated words share one entry. Punctuation is retained.']
    for prompt_id, ids, result in zip(prompt_ids, ids_list, results):
        positions = np.sort(result['differences25'])
        words = original_words(ids, positions, tokenizer)
        total = np.count_nonzero(result['curvature50'] ^ result['attention50'])
        lines.extend(['', f'Prompt {prompt_id}: showing {len(positions)}/{total} disagreeing tokens.'])
        for label, mask in (('Curvature only', result['curvature50']),
                            ('Attention only', result['attention50'])):
            grouped = {}
            for position in positions[mask[positions]]:
                grouped.setdefault(words[int(position)], []).append(str(position))
            entries = '; '.join(f'{json.dumps(word, ensure_ascii=False)} [{", ".join(indices)}]'
                                for word, indices in grouped.items())
            lines.extend([label + ':', textwrap.fill(entries or '(none)', width=110,
                                                    break_long_words=False, break_on_hyphens=False)])
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = json.loads((args.run_root / 'longbench_trec/settings.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(settings['tokenizer_path'],
        cache_dir=settings['cache_dir'], local_files_only=True)
    rng = random.Random(29)
    selections = {name: sorted(rng.sample(range(100), 5)) for name in BENCHMARKS}
    output = args.run_root / 'simple_min_curvature_patterns'
    output.mkdir(exist_ok=True)
    dataset_means = []
    for name in BENCHMARKS:
        run_dir = args.run_root / f'longbench_{name}'
        settings = json.loads((run_dir / 'settings.json').read_text())
        manifest = torch.load(run_dir / 'prompts.pt',
                              map_location='cpu', weights_only=True, mmap=True)
        ids_list = [manifest['input_ids'][i].numpy() for i in selections[name]]
        retention, differences = [], []
        for prompt_id, ids in zip(selections[name], ids_list):
            # Hold only reduced scores for one prompt; map one layer file at a time.
            scores = np.empty((settings['layer_end'], len(ids)), dtype=np.float32)
            for layer in range(settings['layer_end']):
                scores[layer], result = analyze_prompt(
                    run_dir / f'layer_{layer:03d}/prompt_{prompt_id:05d}.pt')
                if layer == 0:
                    differences.append(result)
            retention.append(layer_retention(scores))
            del scores
        retention = np.array(retention)
        write_layer_report(output / f'within_{name}.txt',
                           f'{name.upper()}: LAYER PATTERNS ACROSS FIVE PROMPTS (seed 29)',
                           [f'Prompt {i}' for i in selections[name]], retention)
        write_differences(output / f'layer0_difference25_{name}.txt', name, selections[name],
                          ids_list, differences, tokenizer)
        dataset_means.append(retention.mean(axis=0))
        print(f'{name}: five prompts, {settings["layer_end"]} layers', flush=True)
    write_layer_report(output / 'overall.txt',
                       'LAYER PATTERNS ACROSS DATASETS: FIVE PROMPTS EACH (seed 29)',
                       [name.upper() for name in BENCHMARKS], np.array(dataset_means))
    print(f'Saved reports to {output}', flush=True)


if __name__ == '__main__':
    main()

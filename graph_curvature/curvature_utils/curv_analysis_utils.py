"""Compact A-curvature summaries for individual prompts and saved layer scores."""

from pathlib import Path

import numpy as np
import torch


def finite_cost_statistics(values, chunk_size=65536):
    """Accumulate finite costs in float64 with bounded temporary memory."""
    values = np.asarray(values).reshape(-1)
    total, count = 0.0, 0
    for start in range(0, values.size, chunk_size):
        chunk = values[start:start + chunk_size]
        finite = np.isfinite(chunk)
        total += float(chunk[finite].sum(dtype=np.float64))
        count += int(finite.sum())
    return {'sum': total, 'count': count}


def _summarize_curvatures(curvature):
    """Summarize one prompt's attention-entry curvatures, excluding undefined scores."""
    values = curvature.detach().cpu().numpy() if torch.is_tensor(curvature) else np.asarray(curvature)
    finite = values[np.isfinite(values)]
    return {
        'total_edges': int(values.size),
        'finite_edges': int(finite.size),
        'min_curvature': float(finite.min()) if finite.size else float('nan'),
        'max_curvature': float(finite.max()) if finite.size else float('nan'),
        'negative_curvature': int((finite < 0).sum()),
        'zero_curvature': int((finite == 0).sum()),
        'positive_curvature': int((finite > 0).sum()),
    }


def _write_metrics(stream, metrics, curvature_summary):
    for part in ('prev', 'cur', 'next'):
        stats = metrics['costs'][part]
        mean = stats['sum'] / stats['count'] if stats['count'] else float('nan')
        stream.write(f'{part}_cost_mean: {mean:.8g}\n')
    count = metrics['edge_evaluations']
    for part in ('in', 'out'):
        mean = metrics[f'{part}_neighbor_sum'] / count if count else float('nan')
        stream.write(f'mean_len({part}_neigh): {mean:.8g}\n')
    for name, value in curvature_summary.items():
        stream.write(f'{name}: {value}\n')


def rank_a_curvature(curvature):
    """Rank key indices per head by descending curvature; undefined entries are -1."""
    finite = torch.isfinite(curvature)
    ranking = torch.argsort(
        curvature.masked_fill(~finite, -torch.inf), dim=-1, descending=True, stable=True,
    )
    return ranking.masked_fill(~finite.gather(-1, ranking), -1)


def write_curvature_summary(log_path, prompt_tokens, prompt_records):
    """Keep curvature summaries per prompt; pool only costs, neighbor stats, and counts."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    overall = {
        'costs': {part: {'sum': 0.0, 'count': 0} for part in ('prev', 'cur', 'next')},
        'edge_evaluations': 0, 'in_neighbor_sum': 0, 'out_neighbor_sum': 0,
    }
    with log_path.open('w', encoding='utf-8') as stream:
        stream.write(f'prompt_count: {len(prompt_tokens)}\n')
        stream.write(f'total_token_length: {sum(prompt_tokens)}\n')
        stream.write(f'prompt_token_lengths: {prompt_tokens}\n')
        for sample_idx, (tokens, record) in enumerate(zip(prompt_tokens, prompt_records)):
            stream.write(f'\nprompt {sample_idx}:\ntoken_length: {tokens}\n')
            _write_metrics(stream, record, record['curvature'])
            for part in ('prev', 'cur', 'next'):
                for key in ('sum', 'count'):
                    overall['costs'][part][key] += record['costs'][part][key]
            for key in ('edge_evaluations', 'in_neighbor_sum', 'out_neighbor_sum'):
                overall[key] += record[key]
        stream.write('\noverall:\n')
        counts = {
            name: sum(record['curvature'][name] for record in prompt_records)
            for name in ('total_edges', 'finite_edges', 'negative_curvature',
                         'zero_curvature', 'positive_curvature')
        }
        _write_metrics(stream, overall, counts)
    return log_path

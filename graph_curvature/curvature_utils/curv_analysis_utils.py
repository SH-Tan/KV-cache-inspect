import os
import ast
import numpy as np
import torch


_AGGREGATE_MARKER = "overall:\n"
_PARAMETER_CURVATURE_POINTS = {}
_PARAMETER_OT_NEIGHBOR_POINTS = {}
_PARAMETER_NEIGHBOR_DISTRIBUTION_POINTS = {}
_ACTIVE_PARAMETER_LOGS = set()
_PYPLOT = None


def _analysis_file_path(layer_id, short_name, seq_len, dataset_name, analysis_dir=None):
    if analysis_dir is None:
        analysis_dir = os.path.join(
            os.path.dirname(__file__),
            "curv_analysis",
            f"seq_len_{int(seq_len)}",
            str(dataset_name),
        )
    os.makedirs(analysis_dir, exist_ok=True)
    safe_dataset_name = str(dataset_name).replace("/", "_").replace(" ", "_")
    return os.path.join(
        analysis_dir,
        f"layer_{int(layer_id):03d}_{short_name}_"
        f"{safe_dataset_name}_seq_{int(seq_len):04d}_"
        "min_curv_summary.txt",
    )


def _summarize_curvatures(curvature):
    curvature = np.asarray(curvature)
    finite_mask = np.isfinite(curvature)
    finite_vals = curvature[finite_mask]

    if finite_vals.size == 0:
        return {
            "finite_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
            "min_curv": float("nan"),
            "max_curv": float("nan"),
        }

    return {
        "finite_count": int(finite_vals.size),
        "positive_count": int((finite_vals > 0).sum()),
        "negative_count": int((finite_vals < 0).sum()),
        "zero_count": int((finite_vals == 0).sum()),
        "min_curv": float(finite_vals.min()),
        "max_curv": float(finite_vals.max()),
    }


def _strip_aggregate_footer(content):
    marker_idx = content.find(_AGGREGATE_MARKER)
    if marker_idx != -1:
        return content[:marker_idx].rstrip() + "\n"
    return content


def _write_final_curvature_footer(analysis_path, summary):
    with open(analysis_path, "r", encoding="utf-8") as f:
        content = _strip_aggregate_footer(f.read())

    footer = (
        "\n"
        f"{_AGGREGATE_MARKER}"
        + (
            "source=final_min_curvature_matrix, "
            f"finite_entries={summary['finite_count']}, "
            f"positive_edges={summary['positive_count']}, "
            f"negative_edges={summary['negative_count']}, "
            f"zero_edges={summary['zero_count']}, "
            f"min_curv={summary['min_curv']:.8f}, "
            f"max_curv={summary['max_curv']:.8f}\n"
        )
    )

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.write(footer)


def start_curvature_analysis(
    layer_id,
    short_name,
    sample_idx,
    curvature_shape,
    seq_len,
    dataset_name,
    analysis_dir=None,
):
    analysis_path = _analysis_file_path(
        layer_id,
        short_name,
        seq_len,
        dataset_name,
        analysis_dir=analysis_dir,
    )
    header = (
        f"layer_id: {int(layer_id)}\n"
        f"op_name: {short_name}\n"
        f"dataset_name: {dataset_name}\n"
        f"curvature_shape: {tuple(curvature_shape)}\n"
        f"seq_len: {int(seq_len)}\n"
        "\n"
    )

    if not os.path.exists(analysis_path):
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(header)
    return analysis_path


def append_qproj_cost_statistics(analysis_path, sample_idx, parameter_cost_stats, old_cost_stats, new_cost_stats):
    if analysis_path is None or (
        parameter_cost_stats is None and old_cost_stats is None and new_cost_stats is None
    ):
        return

    def fmt(name, stats):
        if not stats or int(stats.get("count", 0)) <= 0:
            return f"{name}: unavailable\n"
        count = int(stats["count"])
        mean = float(stats["sum"]) / count
        variance = max(float(stats["sum_sq"]) / count - mean * mean, 0.0)
        return (
            f"{name}: "
            f"count={count}, "
            f"mean={mean:.8f}, "
            f"max={float(stats['max']):.8f}, "
            f"min={float(stats['min']):.8f}, "
            f"variance={variance:.8f}\n"
        )

    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write(
            "q_proj_cost_statistics:\n"
            f"sample_idx: {int(sample_idx)}\n"
            + fmt("parameter_cost_1_over_abs_para", parameter_cost_stats)
            + fmt("old_curr_to_next_cost_before_target_scale", old_cost_stats)
            + fmt("new_curr_to_next_cost_after_target_scale", new_cost_stats)
            + "\n"
        )


def _parameter_log_path(log_root, layer_id, short_name, sample_idx, v_idx, u_idx):
    param_name = "down_proj" if short_name == "prev_down_proj" else short_name
    layer_dir = os.path.join(log_root, f"layer_{int(layer_id):03d}", param_name)
    example_dir = os.path.join(layer_dir, f"example_{int(sample_idx):03d}")
    param_dir = os.path.join(example_dir, f"param_v{int(v_idx)}_u{int(u_idx)}")
    os.makedirs(param_dir, exist_ok=True)
    log_path = os.path.join(
        param_dir,
        f"parameter_log_example_{int(sample_idx):03d}.txt",
    )
    return log_path


def prepare_parameter_detail_log(log_root, layer_id, short_name, sample_idx, v_idx, u_idx):
    if log_root is None:
        return None

    log_path = _parameter_log_path(log_root, layer_id, short_name, sample_idx, v_idx, u_idx)
    if log_path in _ACTIVE_PARAMETER_LOGS:
        return log_path

    with open(log_path, "w", encoding="utf-8"):
        pass

    _ACTIVE_PARAMETER_LOGS.add(log_path)
    _PARAMETER_CURVATURE_POINTS.pop(log_path, None)
    _PARAMETER_OT_NEIGHBOR_POINTS.pop(log_path, None)
    _PARAMETER_NEIGHBOR_DISTRIBUTION_POINTS.pop(log_path, None)
    return log_path


def _write_array_block(f, name, value):
    if value is None:
        return
    arr = np.asarray(value, dtype=np.float64)
    with np.printoptions(threshold=np.inf, linewidth=200, precision=8, suppress=False):
        f.write(f"{name}: {arr.tolist()}\n")


def _neighbor_plot_values(edge_res, before_key, reduced_key):
    values = edge_res.get(before_key)
    if values is None:
        values = edge_res.get(reduced_key)
    arr = np.asarray([] if values is None else values, dtype=np.float64).reshape(-1)
    return arr[:-1] if arr.size else arr


def _write_residual_node_value_block(f, name, values):
    if values is None:
        return
    f.write(f"{name}:\n")
    for item in values:
        extra = ""
        if item.get("original_weight_magnitude_to_u") is not None:
            extra += f", original_weight_magnitude_to_u={float(item['original_weight_magnitude_to_u'])}"
        if item.get("node_times_weight_magnitude") is not None:
            extra += f", node_times_weight_magnitude={float(item['node_times_weight_magnitude'])}"
        if item.get("probability_times_node_weight") is not None:
            extra += f", probability_times_node_weight={float(item['probability_times_node_weight'])}"
        if item.get("importance") is not None:
            extra += f", importance={float(item['importance'])}"
        f.write(
            "  "
            f"name={item['name']}, "
            f"node_idx={int(item['node_idx'])}, "
            f"local_idx={int(item['local_idx'])}, "
            f"raw_value={float(item['raw_value'])}, "
            f"normalized_probability={float(item['normalized_probability'])}, "
            f"cost_to_u={float(item['cost_to_u'])}"
            f"{extra}\n"
        )


def _write_out_neighbor_node_value_block(f, values):
    if values is None:
        return
    f.write("out_neighbor_node_values:\n")
    for item in values:
        f.write(
            "  "
            f"node_idx={int(item['node_idx'])}, "
            f"raw_value={float(item['raw_value'])}, "
            f"normalized_probability={float(item['normalized_probability'])}, "
            f"cost_from_source={float(item['cost_from_source'])}\n"
        )


def _write_mu_node_value_block(f, name, values):
    if values is None:
        return
    f.write(f"{name}:\n")
    for item in values:
        parts = []
        for key in (
            "node_type",
            "residual_name",
            "residual_local_idx",
            "node_idx",
            "raw_value",
            "normalized_probability",
            "cost_to_u",
            "original_weight_magnitude_to_u",
            "node_times_weight_magnitude",
            "probability_times_node_weight",
            "importance",
        ):
            if item.get(key) is not None:
                value = item[key]
                if isinstance(value, (int, np.integer)):
                    parts.append(f"{key}={int(value)}")
                elif isinstance(value, (float, np.floating)):
                    parts.append(f"{key}={float(value)}")
                else:
                    parts.append(f"{key}={value}")
        f.write("  " + ", ".join(parts) + "\n")


def append_parameter_detail_log(log_root, layer_id, short_name, edge_res):
    if log_root is None or not edge_res:
        return None

    v_idx = int(edge_res["v_idx"])
    u_idx = int(edge_res["u_idx"])
    sample_idx = int(edge_res["sample_idx"])
    log_path = _parameter_log_path(log_root, layer_id, short_name, sample_idx, v_idx, u_idx)
    seq_idx = int(edge_res["seq_idx"])
    curv = float(edge_res["curv"])
    lpf_curv = edge_res.get("lpf_curv")
    metric_score = edge_res.get("metric_score")
    metric_prev_score = edge_res.get("metric_prev_score")
    metric_next_score = edge_res.get("metric_next_score")
    original_weight_magnitude = edge_res.get("original_weight_magnitude")
    weight_magnitude = edge_res.get("weight_magnitude")
    ot_prev_cost_sum = edge_res.get("ot_prev_score")
    ot_next_cost_sum = edge_res.get("ot_next_score")
    ot_neighbor_cost_sum = edge_res.get("ot_neighbor_score")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"parameter: u={u_idx}, v={v_idx}\n"
            f"parameter_seq: u={u_idx}, v={v_idx}, seq_idx={seq_idx}\n"
            f"sample_idx: {sample_idx}\n"
            f"seq_idx: {seq_idx}\n"
            f"edge_index: ({u_idx}, {v_idx})\n"
            f"len(mu): {edge_res['mu_len']}\n"
            f"len(nu): {edge_res['nu_len']}\n"
            "neighbor_note: mu: in neighbor, nu: out neighbor\n"
            f"W_dist: {edge_res['w_dist']}\n"
            f"sp_uv: {edge_res['sp_uv']}\n"
            f"duv_beta: {edge_res.get('duv_beta', 1.0)}\n"
            f"curv: {curv}\n"
        )
        _write_array_block(f, "in_neighbors", edge_res.get("in_neighbors"))
        _write_array_block(f, "out_neighbors", edge_res.get("out_neighbors"))
        _write_out_neighbor_node_value_block(f, edge_res.get("out_neighbor_node_values"))
        _write_array_block(f, "mu_before_reduction", edge_res.get("mu_before_reduction"))
        _write_array_block(f, "mu", edge_res.get("mu"))
        _write_mu_node_value_block(f, "mu_importance_values", edge_res.get("mu_importance_values"))
        if edge_res.get("mu_node_weighted_sum_to_u") is not None:
            f.write(f"mu_node_weighted_sum_to_u: {float(edge_res['mu_node_weighted_sum_to_u'])}\n")
        _write_array_block(f, "nu_before_reduction", edge_res.get("nu_before_reduction"))
        _write_array_block(f, "nu", edge_res.get("nu"))
        _write_array_block(f, "v_to_out_neighbors_inf_nodes", edge_res.get("v_to_out_neighbors_inf_nodes"))
        if lpf_curv is not None:
            f.write(f"lpf_curv: {float(lpf_curv)}\n")
        if metric_score is not None:
            f.write(f"metric_score: {float(metric_score)}\n")
        if metric_prev_score is not None:
            f.write(f"metric_prev_score: {float(metric_prev_score)}\n")
        if metric_next_score is not None:
            f.write(f"metric_next_score: {float(metric_next_score)}\n")
        if ot_prev_cost_sum is not None:
            f.write(f"ot_prev_score: {float(ot_prev_cost_sum)}\n")
        if ot_next_cost_sum is not None:
            f.write(f"ot_next_score: {float(ot_next_cost_sum)}\n")
        if ot_neighbor_cost_sum is not None:
            f.write(f"ot_neighbor_score: {float(ot_neighbor_cost_sum)}\n")
        if edge_res.get("emd_error") is not None:
            f.write(f"emd_error: {edge_res['emd_error']}\n")
        if original_weight_magnitude is not None:
            f.write(f"original_weight_magnitude: {float(original_weight_magnitude)}\n")
        if weight_magnitude is not None:
            f.write(f"weight_magnitude: {float(weight_magnitude)}\n")
        if edge_res.get("source_node_raw_value") is not None:
            f.write(f"source_node_raw_value: {float(edge_res['source_node_raw_value'])}\n")
        if edge_res.get("source_node_mu_probability") is not None:
            f.write(f"source_node_mu_probability: {float(edge_res['source_node_mu_probability'])}\n")
        if edge_res.get("source_node_times_weight_magnitude") is not None:
            f.write(
                f"source_node_times_weight_magnitude: "
                f"{float(edge_res['source_node_times_weight_magnitude'])}\n"
            )
        if edge_res.get("source_probability_times_node_weight") is not None:
            f.write(
                f"source_probability_times_node_weight: "
                f"{float(edge_res['source_probability_times_node_weight'])}\n"
            )
        _write_residual_node_value_block(
            f,
            "residual_source_node_values",
            edge_res.get("residual_source_node_values"),
        )
        if edge_res.get("v_to_out_neighbors_finite_count") is not None:
            f.write(
                f"v_to_out_neighbors_finite_count: {int(edge_res['v_to_out_neighbors_finite_count'])}\n"
                f"v_to_all_out_finite_count: {int(edge_res['v_to_all_out_finite_count'])}\n"
                f"v_to_all_out_count: {int(edge_res['v_to_all_out_count'])}\n"
            )
        f.write("\n")

    points = _PARAMETER_CURVATURE_POINTS.setdefault(log_path, [])
    points.append((seq_idx, curv))

    neighbor_points = _PARAMETER_NEIGHBOR_DISTRIBUTION_POINTS.setdefault(log_path, [])
    neighbor_points.append((
        seq_idx,
        _neighbor_plot_values(edge_res, "mu_before_reduction", "mu"),
        _neighbor_plot_values(edge_res, "nu_before_reduction", "nu"),
    ))

    if ot_neighbor_cost_sum is not None:
        ot_points = _PARAMETER_OT_NEIGHBOR_POINTS.setdefault(log_path, [])
        ot_points.append(
            (
                seq_idx,
                float("nan") if ot_prev_cost_sum is None else float(ot_prev_cost_sum),
                float("nan") if ot_next_cost_sum is None else float(ot_next_cost_sum),
                float(ot_neighbor_cost_sum),
                curv,
            )
        )

    return log_path


def _refresh_parameter_artifacts(log_path):
    if log_path is None or not os.path.exists(log_path):
        return

    curvature_points = _PARAMETER_CURVATURE_POINTS.get(log_path)
    ot_neighbor_points = _PARAMETER_OT_NEIGHBOR_POINTS.get(log_path)
    neighbor_distribution_points = _PARAMETER_NEIGHBOR_DISTRIBUTION_POINTS.get(log_path)

    write_parameter_metric_statistics(
        log_path,
        metric_points=ot_neighbor_points,
        curvature_points=curvature_points,
    )
    draw_parameter_curvature(log_path, points=curvature_points)
    draw_parameter_neighbor_distribution(log_path, points=neighbor_distribution_points)
    draw_parameter_curvature_lpf_comparison(log_path)
    draw_parameter_ot_neighbor_cost_comparison(log_path, points=ot_neighbor_points)


def _sort_parameter_log_entries(log_path):
    if log_path is None or not os.path.exists(log_path):
        return

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
    if len(raw_blocks) <= 1:
        return

    parsed_blocks = []
    for block in raw_blocks:
        seq_idx = None
        for line in block.splitlines():
            if line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
                break
        if seq_idx is None:
            return
        parsed_blocks.append((seq_idx, block))

    parsed_blocks.sort(key=lambda item: item[0])

    with open(log_path, "w", encoding="utf-8") as f:
        for _, block in parsed_blocks:
            f.write(block)
            f.write("\n\n")


def finalize_parameter_detail_log(log_path):
    _sort_parameter_log_entries(log_path)
    _refresh_parameter_artifacts(log_path)
    _ACTIVE_PARAMETER_LOGS.discard(log_path)


def _read_parameter_curvature_points(log_path):
    points = []
    seq_idx = None

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
            elif line.startswith("curv:") and seq_idx is not None:
                curv = float(line.split(":", 1)[1].strip())
                points.append((seq_idx, curv))
                seq_idx = None

    points.sort(key=lambda item: item[0])
    return points


def _weighted_cost_from_line(line):
    for part in line.split(","):
        part = part.strip()
        if part.startswith("weighted_cost="):
            return float(part.split("=", 1)[1])
    return None


def _read_parameter_ot_neighbor_points(log_path):
    points = []
    seq_idx = None
    curv = None
    prev_cost_sum = None
    next_cost_sum = None
    neighbor_cost_sum = None
    active_block = None

    def maybe_append():
        total = neighbor_cost_sum
        if total is None and (prev_cost_sum is not None or next_cost_sum is not None):
            total = float(prev_cost_sum or 0.0) + float(next_cost_sum or 0.0)
        if seq_idx is not None and curv is not None and total is not None:
            points.append((
                seq_idx,
                float("nan") if prev_cost_sum is None else prev_cost_sum,
                float("nan") if next_cost_sum is None else next_cost_sum,
                total,
                curv,
            ))

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                maybe_append()
                seq_idx = None
                curv = None
                prev_cost_sum = None
                next_cost_sum = None
                neighbor_cost_sum = None
                active_block = None
            elif line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
                active_block = None
            elif line.startswith("curv:"):
                curv = float(line.split(":", 1)[1].strip())
                active_block = None
            elif line.startswith("top_prev_to_target_sum:"):
                prev_cost_sum = float(line.split(":", 1)[1].strip())
                active_block = None
            elif line.startswith("top_next_from_source_sum:"):
                next_cost_sum = float(line.split(":", 1)[1].strip())
                active_block = None
            elif line.startswith("top_neighbor_cost_sum:"):
                neighbor_cost_sum = float(line.split(":", 1)[1].strip())
                active_block = None
            elif line.startswith("ot_prev_score:"):
                prev_cost_sum = float(line.split(":", 1)[1].strip())
                active_block = None
            elif line.startswith("ot_next_score:"):
                next_cost_sum = float(line.split(":", 1)[1].strip())
                active_block = None
            elif line.startswith("ot_neighbor_score:"):
                neighbor_cost_sum = float(line.split(":", 1)[1].strip())
                active_block = None
            elif line == "top_in_neighbors_by_probability_cost_to_target:":
                active_block = "prev"
            elif line == "top_out_neighbors_by_probability_cost_from_source:":
                active_block = "next"
            elif active_block in {"prev", "next"} and "weighted_cost=" in line:
                weighted_cost = _weighted_cost_from_line(line)
                if weighted_cost is None:
                    continue
                if active_block == "prev":
                    prev_cost_sum = float(prev_cost_sum or 0.0) + weighted_cost
                else:
                    next_cost_sum = float(next_cost_sum or 0.0) + weighted_cost
            elif not raw_line.startswith(" "):
                active_block = None

    maybe_append()
    points.sort(key=lambda item: item[0])
    return points


def _read_parameter_lpf_points(log_path):
    points = []
    seq_idx = None
    curv = None
    lpf_curv = None

    def maybe_append():
        if seq_idx is not None and curv is not None and lpf_curv is not None:
            points.append((seq_idx, curv, lpf_curv))

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                maybe_append()
                seq_idx = None
                curv = None
                lpf_curv = None
            elif line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
            elif line.startswith("curv:"):
                curv = float(line.split(":", 1)[1].strip())
            elif line.startswith("lpf_curv:"):
                lpf_curv = float(line.split(":", 1)[1].strip())

    maybe_append()
    points.sort(key=lambda item: item[0])
    return points


def _read_parameter_neighbor_distribution_points(log_path):
    points = []
    seq_idx = None
    mu_before = None
    nu_before = None
    mu = None
    nu = None

    def maybe_append():
        mu_source = mu_before if mu_before is not None else mu
        nu_source = nu_before if nu_before is not None else nu
        if seq_idx is not None and (mu_source is not None or nu_source is not None):
            mu_values = np.asarray([] if mu_source is None else mu_source[:-1], dtype=np.float64)
            nu_values = np.asarray([] if nu_source is None else nu_source[:-1], dtype=np.float64)
            points.append((seq_idx, mu_values, nu_values))

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                maybe_append()
                seq_idx = None
                mu_before = None
                nu_before = None
                mu = None
                nu = None
            elif line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
            elif line.startswith("mu_before_reduction:"):
                mu_before = ast.literal_eval(line.split(":", 1)[1].strip())
            elif line.startswith("nu_before_reduction:"):
                nu_before = ast.literal_eval(line.split(":", 1)[1].strip())
            elif line.startswith("mu:"):
                mu = ast.literal_eval(line.split(":", 1)[1].strip())
            elif line.startswith("nu:"):
                nu = ast.literal_eval(line.split(":", 1)[1].strip())

    maybe_append()
    points.sort(key=lambda item: item[0])
    return points


def _safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() <= 1:
        return float("nan")
    x = x[finite]
    y = y[finite]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_parameter_metric_statistics(log_path, metric_points=None, curvature_points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if metric_points is None:
        metric_points = _PARAMETER_OT_NEIGHBOR_POINTS.get(log_path)
    if metric_points is None:
        metric_points = _read_parameter_ot_neighbor_points(log_path)

    if curvature_points is None:
        curvature_points = _PARAMETER_CURVATURE_POINTS.get(log_path)
    if curvature_points is None:
        curvature_points = _read_parameter_curvature_points(log_path)

    if not metric_points and not curvature_points:
        return None

    stats_path = os.path.splitext(log_path)[0] + "_statistics.txt"

    metric_seq = np.asarray([item[0] for item in metric_points], dtype=np.int64) if metric_points else np.asarray([], dtype=np.int64)
    metric_prev = np.asarray([item[1] for item in metric_points], dtype=np.float64) if metric_points else np.asarray([], dtype=np.float64)
    metric_next = np.asarray([item[2] for item in metric_points], dtype=np.float64) if metric_points else np.asarray([], dtype=np.float64)
    metric_total = np.asarray([item[3] for item in metric_points], dtype=np.float64) if metric_points else np.asarray([], dtype=np.float64)
    metric_curv = np.asarray([item[4] for item in metric_points], dtype=np.float64) if metric_points else np.asarray([], dtype=np.float64)
    curv_only = np.asarray([item[1] for item in curvature_points], dtype=np.float64) if curvature_points else np.asarray([], dtype=np.float64)

    def finite_summary(name, values):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return [
                f"{name}_count: 0",
                f"{name}_min: nan",
                f"{name}_max: nan",
                f"{name}_mean: nan",
            ]
        return [
            f"{name}_count: {int(finite.size)}",
            f"{name}_min: {float(finite.min()):.8f}",
            f"{name}_max: {float(finite.max()):.8f}",
            f"{name}_mean: {float(finite.mean()):.8f}",
        ]

    lines = [
        f"log_path: {log_path}",
        f"unique_seq_count: {int(len(np.unique(metric_seq)) if metric_seq.size else 0)}",
    ]
    lines.extend(finite_summary("curvature", curv_only))
    lines.extend(finite_summary("ot_neighbor_score", metric_total))
    lines.extend(finite_summary("ot_prev_score", metric_prev))
    lines.extend(finite_summary("ot_next_score", metric_next))
    lines.append(f"pearson_ot_neighbor_score_curvature: {_safe_pearson(metric_total, metric_curv):.8f}")
    lines.append(f"pearson_ot_prev_score_curvature: {_safe_pearson(metric_prev, metric_curv):.8f}")
    lines.append(f"pearson_ot_next_score_curvature: {_safe_pearson(metric_next, metric_curv):.8f}")

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return stats_path


def _get_pyplot(log_path):
    global _PYPLOT
    if _PYPLOT is not None:
        return _PYPLOT

    try:
        mpl_config_dir = os.path.join("/tmp", "matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        marker_path = os.path.splitext(log_path)[0] + "_curvature_plot_error.txt"
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"Could not draw curvature plot: {exc}\n")
        return None

    _PYPLOT = plt
    return _PYPLOT


def draw_parameter_curvature(log_path, points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if points is None:
        points = _PARAMETER_CURVATURE_POINTS.get(log_path)
    if points is None:
        points = _read_parameter_curvature_points(log_path)
    if not points:
        return None

    plt = _get_pyplot(log_path)
    if plt is None:
        return None

    ordered_points = sorted(points, key=lambda item: item[0])
    xs = np.asarray([seq_idx for seq_idx, _ in ordered_points], dtype=np.int64)
    ys = np.asarray([curv for _, curv in ordered_points], dtype=np.float64)
    finite = np.isfinite(ys)
    if not finite.any():
        return None

    plot_path = os.path.splitext(log_path)[0] + "_curvature.png"
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.plot(xs[finite], ys[finite], marker="o", linewidth=0.9, markersize=1.8)
    ax.set_xlabel("seq id")
    ax.set_ylabel("curvature")
    ax.set_title(os.path.basename(os.path.dirname(log_path)), fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    return plot_path


def _finite_distribution_rows(points, index):
    rows = []
    for item in points:
        arr = np.asarray(item[index], dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            rows.append(arr)
    return rows


def _distribution_summary(values):
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def _format_distribution_stats(values, label_prefix):
    stats = _distribution_summary(values)
    if stats is None:
        return None
    return (
        f"{label_prefix}: n={stats['count']}\n"
        f"min={stats['min']:.3g}, max={stats['max']:.3g}\n"
        f"mean={stats['mean']:.3g}, median={stats['p50']:.3g}\n"
        f"p10={stats['p10']:.3g}, p50={stats['p50']:.3g}, p90={stats['p90']:.3g}"
    )


def _draw_distribution_stats_text(ax, values, label_prefix):
    text = _format_distribution_stats(values, label_prefix)
    if text is None:
        return
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )


def _representative_distribution_row(rows):
    if not rows:
        return None
    counts = np.asarray([row.size for row in rows], dtype=np.float64)
    median_count = float(np.median(counts))
    row_idx = int(np.argmin(np.abs(counts - median_count)))
    return rows[row_idx]


def _draw_single_seq_distribution(ax, rows, title, label_prefix):
    if not rows:
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)
        return

    values = np.sort(_representative_distribution_row(rows))
    xs = np.arange(1, values.size + 1)
    ax.plot(xs, values, linewidth=1.0, color="tab:blue", label="sorted probabilities")

    percentile_lines = (
        ("min", float(np.min(values)), "tab:gray", "--"),
        ("p10", float(np.percentile(values, 10)), "tab:gray", "--"),
        ("p50", float(np.percentile(values, 50)), "tab:red", "-."),
        ("p90", float(np.percentile(values, 90)), "tab:green", ":"),
        ("max", float(np.max(values)), "tab:gray", "--"),
        ("mean", float(np.mean(values)), "tab:orange", "-"),
        ("median", float(np.median(values)), "tab:purple", ":"),
    )
    for name, value, color, linestyle in percentile_lines:
        ax.axhline(value, linestyle=linestyle, linewidth=0.85, color=color, label=f"{name}={value:.3g}")

    if values.size > 32:
        ax.set_xscale("log")
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size and float(finite_positive.max() / finite_positive.min()) > 100.0:
        ax.set_yscale("log")

    ax.set_title(f"{title} (one representative seq, n={values.size})", fontsize=8)
    ax.set_xlabel("neighbor index sorted small to high")
    ax.set_ylabel("probability")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best", ncol=2)
    _draw_distribution_stats_text(ax, values, label_prefix)


def draw_parameter_neighbor_distribution(log_path, points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if points is None:
        points = _PARAMETER_NEIGHBOR_DISTRIBUTION_POINTS.get(log_path)
    if points is None:
        points = _read_parameter_neighbor_distribution_points(log_path)
    if not points:
        return None

    mu_rows = _finite_distribution_rows(points, 1)
    nu_rows = _finite_distribution_rows(points, 2)
    if not mu_rows and not nu_rows:
        return None

    plt = _get_pyplot(log_path)
    if plt is None:
        return None

    plot_path = os.path.splitext(log_path)[0] + "_neighbor_distribution.png"
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2))
    _draw_single_seq_distribution(
        axes[0],
        mu_rows,
        "in-neighbor mu before reduction",
        "mu",
    )
    _draw_single_seq_distribution(
        axes[1],
        nu_rows,
        "out-neighbor nu before reduction",
        "nu",
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def draw_parameter_ot_neighbor_cost_comparison(log_path, points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if points is None:
        points = _PARAMETER_OT_NEIGHBOR_POINTS.get(log_path)
    if points is None:
        points = _read_parameter_ot_neighbor_points(log_path)
    if not points:
        return None

    plt = _get_pyplot(log_path)
    if plt is None:
        return None

    ordered_points = sorted(points, key=lambda item: item[0])
    xs = np.asarray([item[0] for item in ordered_points], dtype=np.int64)
    prev_sums = np.asarray([item[1] for item in ordered_points], dtype=np.float64)
    next_sums = np.asarray([item[2] for item in ordered_points], dtype=np.float64)
    total_sums = np.asarray([item[3] for item in ordered_points], dtype=np.float64)
    curv = np.asarray([item[4] for item in ordered_points], dtype=np.float64)

    finite_prev = np.isfinite(prev_sums)
    finite_next = np.isfinite(next_sums)
    finite_total = np.isfinite(total_sums)
    finite_corr = finite_total & np.isfinite(curv)
    if not (finite_prev.any() or finite_next.any() or finite_total.any()):
        return None

    corr_text = ""
    if finite_corr.sum() > 1:
        finite_total_values = total_sums[finite_corr]
        finite_curv = curv[finite_corr]
        if np.std(finite_total_values) > 0 and np.std(finite_curv) > 0:
            corr = float(np.corrcoef(finite_total_values, finite_curv)[0, 1])
            corr_text = f", Pearson r={corr:.3f}"

    plot_path = os.path.splitext(log_path)[0] + "_ot_neighbor_cost_curvature_compare.png"
    fig, (ax_seq, ax_scatter) = plt.subplots(2, 1, figsize=(4.6, 4.8))

    if finite_prev.any():
        ax_seq.plot(
            xs[finite_prev],
            prev_sums[finite_prev],
            marker="o",
            linewidth=0.9,
            markersize=1.8,
            label="in-to-v score",
        )
    if finite_next.any():
        ax_seq.plot(
            xs[finite_next],
            next_sums[finite_next],
            marker="s",
            linewidth=0.9,
            markersize=1.8,
            label="u-to-out score",
        )
    if finite_total.any():
        ax_seq.plot(
            xs[finite_total],
            total_sums[finite_total],
            marker="^",
            linewidth=1.1,
            markersize=2.0,
            label="OT neighbor score",
        )
    ax_seq.set_xlabel("seq id")
    ax_seq.set_ylabel("OT neighbor prob * cost")
    ax_seq.grid(True, alpha=0.3)
    ax_seq.legend(fontsize=6, loc="best")

    finite_curv = np.isfinite(curv)
    if finite_curv.any():
        ax_curv = ax_seq.twinx()
        ax_curv.plot(
            xs[finite_curv],
            curv[finite_curv],
            color="tab:red",
            marker="x",
            linewidth=0.9,
            markersize=2.2,
            label="curvature",
        )
        ax_curv.set_ylabel("curvature", color="tab:red")
        ax_curv.tick_params(axis="y", labelcolor="tab:red")

    ax_seq.set_title(
        f"{os.path.basename(os.path.dirname(log_path))}{corr_text}",
        fontsize=8,
    )

    if finite_corr.any():
        ax_scatter.scatter(total_sums[finite_corr], curv[finite_corr], s=10, alpha=0.8)
    ax_scatter.set_xlabel("OT neighbor prob * cost")
    ax_scatter.set_ylabel("curvature")
    ax_scatter.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def draw_parameter_curvature_lpf_comparison(log_path, points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if points is None:
        points = _read_parameter_lpf_points(log_path)
    if not points:
        return None

    plt = _get_pyplot(log_path)
    if plt is None:
        return None

    ordered_points = sorted(points, key=lambda item: item[0])
    xs = np.asarray([item[0] for item in ordered_points], dtype=np.int64)
    raw_curv = np.asarray([item[1] for item in ordered_points], dtype=np.float64)
    lpf_curv = np.asarray([item[2] for item in ordered_points], dtype=np.float64)

    finite_raw = np.isfinite(raw_curv)
    finite_lpf = np.isfinite(lpf_curv)
    if not finite_raw.any() and not finite_lpf.any():
        return None

    plot_path = os.path.splitext(log_path)[0] + "_lpf_curvature_compare.png"
    fig, ax = plt.subplots(figsize=(4.2, 2.6))

    if finite_raw.any():
        ax.plot(
            xs[finite_raw],
            raw_curv[finite_raw],
            marker="o",
            linewidth=0.9,
            markersize=1.8,
            label="raw",
        )
    if finite_lpf.any():
        ax.plot(
            xs[finite_lpf],
            lpf_curv[finite_lpf],
            marker="s",
            linewidth=0.9,
            markersize=1.8,
            label="lpf",
        )

    ax.set_xlabel("seq id")
    ax.set_ylabel("curvature")
    ax.set_title(os.path.basename(os.path.dirname(log_path)), fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def draw_cached_parameter_curvatures(log_root=None):
    drawn_paths = []
    root_path = os.path.abspath(log_root) if log_root is not None else None
    seen_log_paths = set()

    for log_path, points in list(_PARAMETER_CURVATURE_POINTS.items()):
        abs_log_path = os.path.abspath(log_path)
        if root_path is not None and os.path.commonpath([root_path, abs_log_path]) != root_path:
            continue

        seen_log_paths.add(abs_log_path)
        plot_path = draw_parameter_curvature(log_path, points=points)
        if plot_path is not None:
            drawn_paths.append(plot_path)
        stats_path = write_parameter_metric_statistics(
            log_path,
            metric_points=_PARAMETER_OT_NEIGHBOR_POINTS.get(log_path),
            curvature_points=points,
        )
        if stats_path is not None:
            drawn_paths.append(stats_path)
        ot_plot_path = draw_parameter_ot_neighbor_cost_comparison(
            log_path,
            points=_PARAMETER_OT_NEIGHBOR_POINTS.get(log_path),
        )
        if ot_plot_path is not None:
            drawn_paths.append(ot_plot_path)
        neighbor_plot_path = draw_parameter_neighbor_distribution(
            log_path,
            points=_PARAMETER_NEIGHBOR_DISTRIBUTION_POINTS.get(log_path),
        )
        if neighbor_plot_path is not None:
            drawn_paths.append(neighbor_plot_path)
        _PARAMETER_CURVATURE_POINTS.pop(log_path, None)
        _PARAMETER_OT_NEIGHBOR_POINTS.pop(log_path, None)
        _PARAMETER_NEIGHBOR_DISTRIBUTION_POINTS.pop(log_path, None)

    if root_path is not None and os.path.isdir(root_path):
        for current_root, _, file_names in os.walk(root_path):
            for file_name in file_names:
                is_old_log_name = file_name == "parameter_log.txt"
                is_new_log_name = (
                    file_name.startswith("parameter_log_example_")
                    and file_name.endswith(".txt")
                    and not file_name.endswith("_statistics.txt")
                )
                if not (is_old_log_name or is_new_log_name):
                    continue

                log_path = os.path.join(current_root, file_name)
                abs_log_path = os.path.abspath(log_path)
                if abs_log_path in seen_log_paths:
                    continue

                plot_path = draw_parameter_curvature(log_path)
                if plot_path is not None:
                    drawn_paths.append(plot_path)
                stats_path = write_parameter_metric_statistics(log_path)
                if stats_path is not None:
                    drawn_paths.append(stats_path)
                ot_plot_path = draw_parameter_ot_neighbor_cost_comparison(log_path)
                if ot_plot_path is not None:
                    drawn_paths.append(ot_plot_path)
                lpf_plot_path = draw_parameter_curvature_lpf_comparison(log_path)
                if lpf_plot_path is not None:
                    drawn_paths.append(lpf_plot_path)
                neighbor_plot_path = draw_parameter_neighbor_distribution(log_path)
                if neighbor_plot_path is not None:
                    drawn_paths.append(neighbor_plot_path)

    return drawn_paths


def _format_cost_summary_row(name, stats):
    if not stats or int(stats.get("count", 0)) <= 0:
        return (
            f"{name}: count=0, mean=nan, min=nan, max=nan, "
            f"variance=nan, median=nan\n"
        )

    count = int(stats["count"])
    mean = float(stats["sum"]) / count
    variance = max(float(stats["sum_sq"]) / count - mean * mean, 0.0)
    median_count = int(stats.get("median_count", 0))
    median = float(stats["median_sum"]) / median_count if median_count > 0 else float("nan")
    return (
        f"{name}: "
        f"count={count}, "
        f"mean={mean:.8f}, "
        f"min={float(stats['min']):.8f}, "
        f"max={float(stats['max']):.8f}, "
        f"variance={variance:.8f}, "
        f"median={median:.8f}\n"
    )


def append_final_min_curvature_summary(
    analysis_path,
    sample_idx,
    curvature,
    avg_mu_len,
    avg_nu_len,
    curr_dist_finite_edges,
    curr_dist_infinite_edges,
    cost_has_inf,
    cost_inf_count,
    sampled_edge_count=None,
    runtime_sec=None,
    beta_stats=None,
    cost_stats=None,
):
    if torch.is_tensor(curvature):
        curvature = curvature.detach().cpu().numpy()
    else:
        curvature = np.asarray(curvature)

    summary = _summarize_curvatures(curvature)
    finite_entries = int(sampled_edge_count) if sampled_edge_count is not None else summary["finite_count"]
    runtime_sec = float(runtime_sec) if runtime_sec is not None else float("nan")
    beta_stats = beta_stats or {}
    beta_count = int(beta_stats.get("count", 0))
    beta_expected_count = int(beta_stats.get("expected_count", 0))
    source_node_zero_count = int(beta_stats.get("source_node_zero_count", 0))
    target_node_zero_count = int(beta_stats.get("target_node_zero_count", 0))
    zero_weight_parameter_count = int(beta_stats.get("zero_weight_parameter_count", 0))
    if beta_count > 0:
        beta_mean = float(beta_stats["sum"]) / beta_count
        beta_variance = max(float(beta_stats["sum_sq"]) / beta_count - beta_mean * beta_mean, 0.0)
        beta_min = float(beta_stats["min"])
        beta_max = float(beta_stats["max"])
    else:
        beta_mean = float("nan")
        beta_variance = float("nan")
        beta_min = float("nan")
        beta_max = float("nan")
    cost_stats = cost_stats or {}
    prev_cost_row = _format_cost_summary_row("prev_cost", cost_stats.get("prev"))
    cur_cost_row = _format_cost_summary_row("cur_cost", cost_stats.get("cur"))
    next_cost_row = _format_cost_summary_row("next_cost", cost_stats.get("next"))

    with open(analysis_path, "r", encoding="utf-8") as f:
        content = _strip_aggregate_footer(f.read())

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(content)

    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write(
            f"example {int(sample_idx)}: "
            f"curr_dist_finite_edges={int(curr_dist_finite_edges)}, "
            f"curr_dist_infinite_edges={int(curr_dist_infinite_edges)}, "
            f"zero_weight_parameters_skipped={zero_weight_parameter_count}, "
            f"finite_entries={finite_entries}, "
            f"curvature_finite_entries={summary['finite_count']}, "
            f"positive_edges={summary['positive_count']}, "
            f"negative_edges={summary['negative_count']}, "
            f"zero_edges={summary['zero_count']}, "
            f"avg_len(mu)={float(avg_mu_len):.8f}, "
            f"avg_len(nu)={float(avg_nu_len):.8f}, "
            f"cost_has_inf={bool(cost_has_inf)}, "
            f"cost_inf_count={int(cost_inf_count)}, "
            f"runtime_sec={float(runtime_sec):.6f}, "
            f"min_curv={summary['min_curv']:.8f}, "
            f"max_curv={summary['max_curv']:.8f}\n"
            f"{prev_cost_row}"
            f"{cur_cost_row}"
            f"{next_cost_row}"
            f"beta: "
            f"count={beta_count}, "
            f"expected_count={beta_expected_count}, "
            f"source_node_zero_count={source_node_zero_count}, "
            f"target_node_zero_count={target_node_zero_count}, "
            f"mean={beta_mean:.8f}, "
            f"min={beta_min:.8f}, "
            f"max={beta_max:.8f}, "
            f"variance={beta_variance:.8f}\n\n"
        )


def append_final_curvature_overall(layer_id, short_name, curvature, seq_len, dataset_name, analysis_dir=None):
    if torch.is_tensor(curvature):
        curvature = curvature.detach().cpu().numpy()
    else:
        curvature = np.asarray(curvature)

    analysis_path = _analysis_file_path(
        layer_id,
        short_name,
        seq_len,
        dataset_name,
        analysis_dir=analysis_dir,
    )
    if not os.path.exists(analysis_path):
        header = (
            f"layer_id: {int(layer_id)}\n"
            f"op_name: {short_name}\n"
            f"dataset_name: {dataset_name}\n"
            f"curvature_shape: {tuple(curvature.shape)}\n"
            f"seq_len: {int(seq_len)}\n"
            "\n"
        )
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(header)

    summary = _summarize_curvatures(curvature)
    _write_final_curvature_footer(analysis_path, summary)

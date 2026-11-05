from graph_curvature.curvature_utils.curv_distribution_utils import _resolve_node_name

GRAPH = {
    "q_proj": {"layer": 1, "prev": [], "next": ["A"], "prev_in": [], "next_out": ["Att_out"]},
    "k_proj": {"layer": 1, "prev": [], "next": ["A"], "prev_in": [], "next_out": ["Att_out"]},
    "v_proj": {"layer": 1, "prev": [], "next": ["A"], "prev_in": [], "next_out": ["Att_out"]},
    "A": {"layer": 2, "prev": ["v_proj"], "next": ["o_proj"], "prev_in": ["layer_input"], "next_out": ["o_proj"]},
    "o_proj": {"layer": 3, "prev": ["A"], "next": [], "prev_in": [], "next_out": []},
}


def _resolve_graph_sets(operations, short_name):
    rel = GRAPH.get(short_name, {})

    prev_in_name = _resolve_node_name(operations, rel.get("prev_in", []))
    next_out_name = _resolve_node_name(operations, rel.get("next_out", []))
    prev_cost_names = rel.get("prev", [])
    next_cost_names = rel.get("next", [])
    residual_names = [name for name in rel.get("residual", []) if name in operations]

    return {
        "prev_in": operations.get(prev_in_name),
        "next_out": operations.get(next_out_name),
        "prev_in_name": prev_in_name,
        "next_out_name": next_out_name,
        "prev_cost_names": prev_cost_names,
        "next_cost_names": next_cost_names,
        "residual_names": residual_names,
    }
    
    
# GRAPH = {
#     "prev_down_proj": {"layer": 0, "prev": ["prev_up_proj", "prev_gate_proj"], "next": ["v_proj"], "prev_in": ["prev_o_proj"], "next_out": ["v_proj"]},
#     "q_proj": {"layer": 1, "prev": ["prev_down_proj"], "next": ["o_proj"], "prev_in": ["prev_gate_up_out"], "next_out": ["Att_out"]},
#     "k_proj": {"layer": 1, "prev": ["prev_down_proj"], "next": ["o_proj"], "prev_in": ["prev_gate_up_out"], "next_out": ["Att_out"]},
#     "v_proj": {"layer": 1, "prev": ["prev_down_proj"], "next": ["o_proj"], "prev_in": ["prev_gate_up_out"], "next_out": ["Att_out"]},
#     "A": {"layer": 2, "prev": [], "next": ["v_proj"], "prev_in": [], "next_out": ["v_proj"]},
#     "o_proj": {"layer": 3, "prev": ["q_proj", "k_proj", "v_proj"], "next": ["gate_proj", "up_proj"], "prev_in": ["layer_input"], "next_out": ["gate_proj", "up_proj"]},
#     "gate_proj": {"layer": 4.1, "prev": ["o_proj"], "next": ["down_proj"], "prev_in": ["Att_out"], "next_out": ["down_proj"]},
#     "up_proj": {"layer": 4.2, "prev": ["o_proj"], "next": ["down_proj"], "prev_in": ["Att_out"], "next_out": ["down_proj"]},
#     "down_proj": {"layer": 5, "prev": ["up_proj", "gate_proj"], "next": ["v_proj"], "prev_in": ["o_proj"], "next_out": ["lm_head"]},
#     "lm_head": {"layer": 6, "prev": ["down_proj"], "next": [], "prev_in": ["gate_up_out"], "next_out": []},
# }

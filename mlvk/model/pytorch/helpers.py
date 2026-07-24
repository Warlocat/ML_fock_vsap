"""Small training and inference helpers for the production GNN."""

import gc
import random

import numpy as np
import torch
import torch.nn as nn

from mlvk.model.pytorch.data import reconstruct_rank3
from mlvk.model.pytorch.pt_alias import get_model_from_alias


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_graph_ensemble_mean(values):
    return np.mean(values, axis=0)


def get_graph_ensemble_uncertainty(values, values_mean=None):
    if values_mean is None:
        values_mean = get_graph_ensemble_mean(values)
    return (np.sum((values - values_mean) ** 2, axis=0) / (values.shape[0] - 1)) ** 0.5


def load_pt_model(params, model_name, **kwargs):
    model = get_model_from_alias(model_name, **kwargs)
    model.load_state_dict(params, strict=kwargs.get("load_strict", True))
    model.eval()
    return model


def predict_wrapper_graph(graph, model_state, model_alias, **kwargs):
    if model_alias != "GNN_Multiblock_faster":
        raise ValueError(f"Unsupported model alias: {model_alias}")
    model = load_pt_model(model_state, model_alias, **kwargs)
    model.to(kwargs.get("device", "cpu"))
    block_output = reconstruct_rank3(graph.output_scaffold, graph.nmo)
    with torch.no_grad():
        value = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.node_indices_nonzero,
            graph.edge_indices_nonzero,
            graph.nmo,
            graph.nout,
            graph.inds_val,
            graph.inds_core,
            block_output,
            mol=graph,
        ).detach().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return value


def predict_wrapper_graph_ensemble(graph, model_states, model_alias, **kwargs):
    values = [
        predict_wrapper_graph(graph, state, model_alias, **kwargs)
        for state in model_states
    ]
    return torch.stack(values).numpy()

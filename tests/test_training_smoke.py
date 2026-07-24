import torch

from mlvk.model.pytorch.data import Graph
from mlvk.model.pytorch.pt_alias import get_model_from_alias
from mlvk.model.pytorch.train import train_multiblock_model_fast


def test_one_epoch_two_block_training():
    torch.set_default_dtype(torch.float64)
    graph = Graph(
        x=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        edge_attr=torch.randn(4, 2),
    )
    graph.node_indices_nonzero = torch.arange(4)
    graph.edge_indices_nonzero = graph.edge_index.T.clone()
    graph.vk_ii = torch.randn(4, 1)
    graph.vk_ij = torch.randn(4, 1)
    graph.output_scaffold = torch.zeros(16, 1)
    graph.inds_core = torch.tensor([0, 1])
    graph.inds_val = torch.tensor([2, 3])
    graph.nmo = 4
    graph.nout = 1

    block = {
        "n_out": 1,
        "node_input": 2,
        "edge_input": 2,
        "node_nhidden": 4,
        "edge_nhidden": 4,
        "mpl_channels": 3,
        "message_passing_updates": 1,
        "precision": "double",
    }
    model = get_model_from_alias(
        "GNN_Multiblock_faster",
        scale=0.01,
        **{
            "core-core_model_kwargs": dict(block),
            "valence-valence_model_kwargs": dict(block),
        },
    )
    trained = train_multiblock_model_fast(
        [graph],
        model,
        {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "subgraph_learning": True,
            "init_loss_epoch": 10,
            "shuffle": False,
        },
        compiles=False,
    )
    assert trained is model

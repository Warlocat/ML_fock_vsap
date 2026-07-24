"""Model aliases retained for the published ML-Fock workflow."""

from mlvk.model.pytorch.gnn_multiblock_faster import GNN_Multiblock_faster


def get_model_from_alias(alias, **kwargs):
    """Construct the production model from its serialized alias."""
    if alias != "GNN_Multiblock_faster":
        raise ValueError(
            f"Unsupported model alias {alias!r}; this release retains only "
            "'GNN_Multiblock_faster'."
        )
    return GNN_Multiblock_faster(**kwargs)

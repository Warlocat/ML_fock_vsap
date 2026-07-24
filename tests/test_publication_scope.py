from pathlib import Path


def test_experimental_fragment_paths_are_absent():
    root = Path(__file__).parents[1] / "mlvk"
    source = "\n".join(path.read_text() for path in root.rglob("*.py"))
    assert "fragment_id" not in source
    assert "beta_block_e" not in source
    assert "beta_block_j" not in source


def test_only_production_model_alias_is_exposed():
    from mlvk.model.pytorch.pt_alias import get_model_from_alias

    try:
        get_model_from_alias("GNN_Fourier")
    except ValueError as error:
        assert "GNN_Multiblock_faster" in str(error)
    else:
        raise AssertionError("An excluded architecture was accepted")

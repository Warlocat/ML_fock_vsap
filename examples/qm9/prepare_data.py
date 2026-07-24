"""Build and serialize the graph dataset described by a JSON configuration."""

import argparse
import json
from pathlib import Path

from mlvk.model.gnn_model import GNN_Model
from mlvk.model.preprocessing import DataScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())

    basis_name = cfg["basis"].split("_")[0]
    node_scale = cfg.get(
        "scaler_applies_ii",
        [i for i, name in enumerate(cfg["feature_list_ii"]) if basis_name in name],
    )
    edge_scale = cfg.get(
        "scaler_applies_ij",
        [
            i
            for i, name in enumerate(cfg["feature_list_ij"])
            if basis_name in name or name == "atomic_distance"
        ],
    )
    model = GNN_Model(
        cfg["feature_list_ii"],
        cfg["feature_list_ij"],
        cfg["targets"],
        cfg["torch_data_root"],
        transformer=DataScaler(applies_to_ii=node_scale, applies_to_ij=edge_scale),
        exclude_core=cfg.get("exclude_core", False),
        edge_cutoff=cfg.get("edge_cutoff", 0.0),
        edge_cutoff_features=cfg.get("edge_cutoff_features"),
        basis_set=cfg["basis_set"],
        basis=cfg["basis"],
        custom_minao_path=cfg.get("custom_minao_path"),
        downfold_basis_set=cfg.get("downfold_basis_set", "minao"),
        blockdiag_on=cfg.get("blockdiag_on", "hcore"),
        sap_type=cfg.get("sap_type"),
        precision=cfg.get("precision", "double"),
    )
    model = model.load_dset_mpi(cfg["train_files"])
    output = Path(cfg["gnn_model_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    model.dump(output)


if __name__ == "__main__":
    main()

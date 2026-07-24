"""Train the production two-block GNN from a prepared graph dataset."""

import argparse
import json
from pathlib import Path

import joblib
import torch

from mlvk.model.pytorch.pt_alias import get_model_from_alias
from mlvk.model.pytorch.train import train_multiblock_model_fast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())

    wrapper = joblib.load(cfg["gnn_model_file"])
    wrapper.idx_fock_sum = cfg["model_kwargs"].get("idx_fock_sum", [0])
    network = get_model_from_alias(cfg["model_alias"], **cfg["model_kwargs"])
    network = train_multiblock_model_fast(
        wrapper.data,
        network,
        cfg["train_config"],
        loss_kwargs=cfg["loss_kwargs"],
        compiles=False,
        read_chkpt=args.checkpoint,
        save_chkpt=args.save_checkpoints,
    )

    wrapper.model_states = [network.state_dict()]
    wrapper.model_kwargs = cfg["model_kwargs"]
    wrapper.model_alias = cfg["model_alias"]
    wrapper.dump(cfg["gnn_model_file"])
    torch.save(network.state_dict(), str(cfg["gnn_model_file"]).replace(".joblib", "_wts.pt"))


if __name__ == "__main__":
    main()

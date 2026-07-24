"""Training loop used for the two-block molecular ML-Fock model."""

import time
import warnings

import numpy as np
import torch
import torch.optim as optim
import torch_geometric
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader

from mlvk.model.pytorch.data import (
    construct_vk_from_graph,
    get_subgraph,
    reconstruct_rank2,
    reconstruct_rank3,
)
from mlvk.model.pytorch.helpers import count_parameters, setup_seed
from mlvk.model.pytorch.loss import VKLoss


def compute_block_loss(mol, vk_ml, vk_true, criterion, do_extra_penalties=True):
    mo_energy = getattr(mol, "mo_energy", None)
    dm = getattr(mol, "dm", None)
    prev_fock = getattr(mol, "prev_fock", None)
    dipoles = [getattr(mol, f"dipole_{axis}", None) for axis in "xyz"]

    if mo_energy is not None:
        mo_energy = mo_energy / criterion.scale
    if dm is not None:
        dm = reconstruct_rank2(dm, mol.nmo)
    if prev_fock is not None:
        prev_fock = reconstruct_rank2(prev_fock, mol.nmo) / criterion.scale
    if dipoles[0] is not None:
        dipoles = [reconstruct_rank2(value, mol.nmo) for value in dipoles]
    else:
        dipoles = None

    return criterion(
        vk_ml,
        vk_true,
        mo_energy=mo_energy,
        dm=dm,
        nocc=getattr(mol, "nocc", None),
        dipoles=dipoles,
        prev_fock=prev_fock,
        do_extra_penalties=do_extra_penalties,
    )


def _step_schedulers(epoch, schedulers, train_config):
    ranges = (
        train_config.get("cyclic_range", [-1, -2]),
        train_config.get("exp_range", [-1, -2]),
        train_config.get("cos_range", [-1, -2]),
    )
    for scheduler, active_range in zip(schedulers, ranges):
        if active_range[0] <= epoch < active_range[1]:
            scheduler.step()


def train_multiblock_model_fast(
    data,
    model,
    train_config,
    seed=42,
    loss_kwargs=None,
    rank=0,
    debug_interval=None,
    maxnmo=1000,
    compiles=False,
    read_chkpt=None,
    save_chkpt=False,
    chkpt_interval=50,
    **_,
):
    """Train core-core and valence-valence GNN blocks.

    Extra MO/density/dipole penalties are applied only to block indices listed
    in ``train_config['loss_blocks']`` after ``init_loss_epoch``.
    """
    setup_seed(seed)
    torch_geometric.seed.seed_everything(seed)
    device = train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but unavailable; using CPU")
        device = "cpu"

    model.maxnmo = maxnmo
    model = model.to(device)
    if compiles:
        model = torch.compile(model, dynamic=True)
    model.train()

    epochs = int(train_config.get("epochs", 1000))
    learning_rate = train_config.get("learning_rate", 1e-3)
    betas = train_config.get("betas", [0.9, 0.999])
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=train_config.get("weight_decay", 0.0),
        betas=tuple(betas),
        eps=train_config.get("eps", 1e-8),
    )

    cyclic_range = train_config.get("cyclic_range", [-1, -2])
    cyclic = CyclicLR(
        optimizer,
        base_lr=train_config.get("cycle_lr_min", learning_rate),
        max_lr=train_config.get("cycle_lr_max", learning_rate),
        step_size_up=max(1, int((cyclic_range[1] - cyclic_range[0]) / (2 * train_config.get("lr_cycles", 1.0)))),
        cycle_momentum=False,
    )
    exponential = ExponentialLR(
        optimizer, gamma=train_config.get("learning_rate_decay_rate", 1.0)
    )
    cosine = CosineAnnealingWarmRestarts(
        optimizer, train_config.get("cosine_t0", epochs)
    )
    schedulers = (cyclic, exponential, cosine)

    criterion_kwargs = dict(loss_kwargs or {})
    criterion_kwargs["scale"] = getattr(model, "scale", 1.0)
    criterion = VKLoss(**criterion_kwargs).to(device)

    batch_size = min(int(train_config.get("batch_size", len(data))), len(data))
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=train_config.get("shuffle", True),
        pin_memory=train_config.get("pin_memory", False),
        num_workers=train_config.get("num_workers", 0),
    )
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = train_config.get("prefetch_factor", 2)
        loader_kwargs["persistent_workers"] = train_config.get("persistent_workers", False)
    loader = DataLoader(data, **loader_kwargs)

    training_scale = getattr(model, "scale", 1.0)
    init_loss_epoch = train_config.get("init_loss_epoch", 0)
    loss_blocks = set(train_config.get("loss_blocks", [1]))
    if not train_config.get("subgraph_learning", True):
        raise ValueError("This release supports the reported subgraph-learning workflow only")

    if rank == 0:
        print("train_config:", train_config, flush=True)
        print("loss_kwargs:", criterion_kwargs, flush=True)
        print("device:", device, flush=True)
        print("parameters:", count_parameters(model), flush=True)

    epoch_start = 0
    if read_chkpt is not None:
        checkpoint = torch.load(read_chkpt, map_location=device, weights_only=False)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            for name, scheduler in zip(("cyclic", "exponential", "cosine"), schedulers):
                if name in checkpoint:
                    scheduler.load_state_dict(checkpoint[name])
            epoch_start = checkpoint.get("epoch", -1) + 1
        else:
            model.load_state_dict(checkpoint)

    for epoch in range(epoch_start, epochs):
        started = time.time()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = torch.zeros((), dtype=torch.float64, device=device)

            for graph_index in range(batch.num_graphs):
                mol = batch[graph_index]
                vk_true = construct_vk_from_graph(mol.vk_ii, mol.vk_ij, mol) / training_scale
                index_tuple = (mol.inds_core, mol.inds_val)
                for block_index in range(model.nblocks):
                    rows_id = int(model.index_block_combinations[block_index, 0])
                    cols_id = int(model.index_block_combinations[block_index, 1])
                    rows, cols = index_tuple[rows_id], index_tuple[cols_id]
                    mol_block = get_subgraph(mol, rows)
                    scaffold = reconstruct_rank3(mol_block.output_scaffold, mol_block.nmo)
                    vk_block = model(
                        mol_block.x,
                        mol_block.edge_index,
                        mol_block.edge_attr,
                        mol_block.node_indices_nonzero,
                        mol_block.edge_indices_nonzero,
                        mol_block.nmo,
                        mol_block.nout,
                        mol_block.inds_core,
                        mol_block.inds_val,
                        scaffold,
                        this_ib=block_index,
                    )
                    vk_ml = vk_true.clone().to(dtype=vk_block.dtype)
                    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
                    vk_ml[rr, cc] = vk_block[rr, cc]
                    vk_ml[cc, rr] = vk_block[cc, rr]
                    use_extras = epoch >= init_loss_epoch and block_index in loss_blocks
                    loss = loss + compute_block_loss(
                        mol_block, vk_ml, vk_true, criterion, use_extras
                    )

            loss.backward()
            optimizer.step()

        _step_schedulers(epoch, schedulers, train_config)
        if rank == 0:
            elapsed = time.time() - started
            print(f"Epoch {epoch:04d}: loss={loss.item():.6e}, time={elapsed:.2f}s", flush=True)

        if save_chkpt and epoch % chkpt_interval == 0 and rank == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cyclic": cyclic.state_dict(),
                    "exponential": exponential.state_dict(),
                    "cosine": cosine.state_dict(),
                    "epoch": epoch,
                },
                f"checkpoint_epoch_{epoch}.pt",
            )

    return model._orig_mod if compiles else model

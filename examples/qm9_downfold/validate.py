"""Validate downfolded ML-Fock models and write machine-readable CSV results."""

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd
from mpi4py import MPI

from mlvk.workflow.get_ml_info import validate_model_mol_downfold


def validate_one(checkpoint, model_file):
    result = validate_model_mol_downfold(checkpoint, model_file)
    result["chkfile"] = checkpoint
    result["model_file"] = model_file
    for key in ("mo_energy_ml", "lowdin_charge_ml", "lowdin_charge_true", "dm_ml"):
        result.pop(key, None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path, help="JSON with chkfiles and model_files arrays")
    parser.add_argument("output", type=Path, help="Destination CSV")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        spec = json.loads(args.spec.read_text())
        tasks = list(product(spec["chkfiles"], spec["model_files"]))
    else:
        tasks = None
    tasks = comm.bcast(tasks, root=0)

    local_results = []
    for checkpoint, model_file in tasks[comm.rank :: comm.size]:
        if not Path(checkpoint).exists():
            continue
        try:
            local_results.append(validate_one(checkpoint, model_file))
        except Exception as error:
            local_results.append(
                {"chkfile": checkpoint, "model_file": model_file, "error": str(error)}
            )

    gathered = comm.gather(local_results, root=0)
    if comm.rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result for group in gathered for result in group).to_csv(
            args.output, index=False
        )


if __name__ == "__main__":
    main()

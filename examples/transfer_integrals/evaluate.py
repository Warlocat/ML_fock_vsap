"""Evaluate ML and reference transfer integrals for one dimer geometry."""

import argparse
import tempfile
from pathlib import Path

import joblib
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf.data.nist import HARTREE2EV

from mlvk.workflow.generate import do_rks_calculation
from mlvk.workflow.transfer_integrals import effective_hamiltonian, summarize_transfer


def read_fragments(path, atoms_fragment1, basis):
    lines = path.read_text().splitlines()
    atom_count = int(lines[0])
    atoms = [line.split()[:4] for line in lines[2 : 2 + atom_count]]
    if not 0 < atoms_fragment1 < atom_count:
        raise ValueError("--atoms-fragment1 must split the XYZ into two nonempty fragments")

    def build(selected):
        mol = gto.M(atom=[(a[0], tuple(map(float, a[1:]))) for a in selected], basis=basis)
        return mol

    return build(atoms), build(atoms[:atoms_fragment1]), build(atoms[atoms_fragment1:])


def frontier_indices(mol, carrier, degeneracy):
    nocc = mol.nelectron // 2
    if carrier == "hole":
        return list(range(nocc - degeneracy, nocc))
    return list(range(nocc, nocc + degeneracy))


def predict_fock(mol, model, checkpoint, functional):
    mf, _ = do_rks_calculation(mol, str(checkpoint), xc=functional, verbose=0)
    correction = model.predict_vk(str(checkpoint))[:, :, getattr(model, "idx_fock", 0)]
    basis_name = model.basis.split("_")[0]
    if "delta" in model.targets[0]:
        correction = correction + model.pdata[f"fock_{basis_name}"]
    c_lo_ao = model.pdata[f"C_{basis_name}_ao"]
    return mf, np.linalg.multi_dot((c_lo_ao.T, correction, c_lo_ao))


def report(label, mo1, mo2, fock, overlap):
    h_eff, _ = effective_hamiltonian(mo1, mo2, fock, overlap)
    summary = summarize_transfer(h_eff)
    print(f"{label}:")
    print(f"  average splitting: {summary['average_splitting'] * HARTREE2EV:.8f} eV")
    print(f"  site asymmetry:    {summary['site_energy_asymmetry'] * HARTREE2EV:.8f} eV")
    print(f"  |2J| Frobenius:    {summary['two_j_frobenius'] * HARTREE2EV:.8f} eV")
    singular = summary["two_j_singular_values"] * HARTREE2EV
    print("  |2J| singular:     " + ", ".join(f"{value:.8f}" for value in singular) + " eV")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--atoms-fragment1", type=int, required=True)
    parser.add_argument("--basis", default="ccpvdz")
    parser.add_argument("--functional", default="camb3lyp")
    parser.add_argument("--carrier", choices=("hole", "particle"), default="hole")
    parser.add_argument("--degeneracy", type=int, default=2)
    args = parser.parse_args()

    dimer, monomer1, monomer2 = read_fragments(args.geometry, args.atoms_fragment1, args.basis)
    indices1 = frontier_indices(monomer1, args.carrier, args.degeneracy)
    indices2 = frontier_indices(monomer2, args.carrier, args.degeneracy)
    model = joblib.load(args.model)

    with tempfile.TemporaryDirectory(prefix="ml_fock_transfer_") as tmp:
        tmp = Path(tmp)
        mf1, fock_ml1 = predict_fock(monomer1, model, tmp / "monomer1.chk", args.functional)
        mf2, fock_ml2 = predict_fock(monomer2, model, tmp / "monomer2.chk", args.functional)
        mfd, fock_mld = predict_fock(dimer, model, tmp / "dimer.chk", args.functional)

        mo1_ref = mf1.mo_coeff[:, indices1]
        mo2_ref = mf2.mo_coeff[:, indices2]
        _, mo1_ml_all = scipy.linalg.eigh(fock_ml1, mf1.get_ovlp())
        _, mo2_ml_all = scipy.linalg.eigh(fock_ml2, mf2.get_ovlp())
        report("reference", mo1_ref, mo2_ref, mfd.get_fock(), mfd.get_ovlp())
        report("ML", mo1_ml_all[:, indices1], mo2_ml_all[:, indices2], fock_mld, mfd.get_ovlp())


if __name__ == "__main__":
    main()

"""Gauge-invariant transfer-integral analysis in frontier-orbital subspaces."""

import numpy as np


def symmetric_inverse_sqrt(matrix, threshold=1e-10):
    """Return the symmetric inverse square root of a positive overlap matrix."""
    values, vectors = np.linalg.eigh(matrix)
    if np.min(values) <= threshold:
        raise np.linalg.LinAlgError(
            f"Frontier-space overlap is singular (smallest eigenvalue={np.min(values):.3e})"
        )
    return (vectors / np.sqrt(values)) @ vectors.conj().T


def effective_hamiltonian(mo1, mo2, fock_dimer, overlap_dimer):
    """Build and Löwdin-orthogonalize the two-fragment effective Hamiltonian.

    ``mo1`` and ``mo2`` have shapes ``(nao_fragment, nfrontier)``. Both
    fragments must use the same frontier-subspace dimension. Dimension one is
    the conventional 2x2 analysis; dimension two gives the gauge-invariant
    4x4 treatment used for benzene's degenerate HOMO pair.
    """
    mo1 = np.asarray(mo1)
    mo2 = np.asarray(mo2)
    n1, width1 = mo1.shape
    n2, width2 = mo2.shape
    if width1 != width2:
        raise ValueError("The two frontier subspaces must have equal dimensions")
    if fock_dimer.shape != (n1 + n2, n1 + n2):
        raise ValueError("Dimer Fock dimensions do not match the monomer AO dimensions")

    f11 = fock_dimer[:n1, :n1]
    f21 = fock_dimer[n1:, :n1]
    f22 = fock_dimer[n1:, n1:]
    s11 = overlap_dimer[:n1, :n1]
    s21 = overlap_dimer[n1:, :n1]
    s22 = overlap_dimer[n1:, n1:]

    e1 = mo1.conj().T @ f11 @ mo1
    e2 = mo2.conj().T @ f22 @ mo2
    coupling = mo2.conj().T @ f21 @ mo1
    overlap1 = mo1.conj().T @ s11 @ mo1
    overlap2 = mo2.conj().T @ s22 @ mo2
    overlap12 = mo2.conj().T @ s21 @ mo1

    h_eff = np.block([[e1, coupling.conj().T], [coupling, e2]])
    s_eff = np.block(
        [[overlap1, overlap12.conj().T], [overlap12, overlap2]]
    )
    s_inverse_sqrt = symmetric_inverse_sqrt(s_eff)
    return s_inverse_sqrt @ h_eff @ s_inverse_sqrt, s_eff


def summarize_transfer(orthogonal_hamiltonian):
    """Return basis-invariant splitting and transfer-coupling summaries in Hartree."""
    h_eff = np.asarray(orthogonal_hamiltonian)
    if h_eff.ndim != 2 or h_eff.shape[0] != h_eff.shape[1] or h_eff.shape[0] % 2:
        raise ValueError("Expected an even-dimensional square effective Hamiltonian")
    width = h_eff.shape[0] // 2
    eigenvalues = np.linalg.eigvalsh(h_eff)
    coupling = h_eff[width:, :width]
    singular_values = np.linalg.svd(coupling, compute_uv=False)
    return {
        "eigenvalues": eigenvalues,
        "average_splitting": np.mean(eigenvalues[width:]) - np.mean(eigenvalues[:width]),
        "site_energy_asymmetry": abs(np.trace(h_eff[:width, :width]) - np.trace(h_eff[width:, width:])) / width,
        "two_j_frobenius": 2.0 * np.linalg.norm(coupling, "fro"),
        "two_j_singular_values": 2.0 * singular_values,
    }

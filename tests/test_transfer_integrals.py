import numpy as np

from mlvk.workflow.transfer_integrals import effective_hamiltonian, summarize_transfer


def test_two_by_two_limit():
    fock = np.array([[-0.30, 0.012], [0.012, -0.28]])
    h_eff, overlap = effective_hamiltonian(
        np.ones((1, 1)), np.ones((1, 1)), fock, np.eye(2)
    )
    summary = summarize_transfer(h_eff)
    np.testing.assert_allclose(overlap, np.eye(2))
    np.testing.assert_allclose(h_eff, fock)
    np.testing.assert_allclose(summary["two_j_frobenius"], 0.024)


def test_degenerate_subspace_is_gauge_invariant():
    e1 = np.diag([-0.31, -0.30])
    e2 = np.diag([-0.29, -0.28])
    coupling = np.array([[0.010, -0.004], [0.006, 0.008]])
    fock = np.block([[e1, coupling.T], [coupling, e2]])
    overlap = np.eye(4)

    h_reference, _ = effective_hamiltonian(np.eye(2), np.eye(2), fock, overlap)
    angle1, angle2 = 0.37, -0.61
    rotation1 = np.array([[np.cos(angle1), -np.sin(angle1)], [np.sin(angle1), np.cos(angle1)]])
    rotation2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    h_rotated, _ = effective_hamiltonian(rotation1, rotation2, fock, overlap)

    reference = summarize_transfer(h_reference)
    rotated = summarize_transfer(h_rotated)
    np.testing.assert_allclose(rotated["eigenvalues"], reference["eigenvalues"], atol=1e-14)
    np.testing.assert_allclose(rotated["two_j_frobenius"], reference["two_j_frobenius"], atol=1e-14)
    np.testing.assert_allclose(
        rotated["two_j_singular_values"], reference["two_j_singular_values"], atol=1e-14
    )

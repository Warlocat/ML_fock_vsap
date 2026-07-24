import numpy as np

from mlvk.workflow.ham_downfolding import (
    get_X_matrix,
    get_downfold_matrix,
    get_downfold_matrix_extra,
)


def test_downfolded_hamiltonian_recovers_selected_eigenvalues():
    rng = np.random.default_rng(7)
    trial = rng.normal(size=(5, 5))
    hamiltonian = trial + trial.T
    selected = [0, 1, 2]
    x_matrix, returned = get_X_matrix(hamiltonian, 3, idx_mo=selected)
    effective = get_downfold_matrix(hamiltonian, x_matrix)

    expected = np.linalg.eigvalsh(hamiltonian)[selected]
    np.testing.assert_array_equal(returned, selected)
    np.testing.assert_allclose(np.linalg.eigvalsh(effective), expected, atol=1e-10)


def test_extra_components_sum_to_complete_downfolding():
    rng = np.random.default_rng(11)
    trial = rng.normal(size=(6, 6))
    hamiltonian = trial + trial.T
    x_matrix, _ = get_X_matrix(hamiltonian, 3, idx_mo=[0, 1, 2])
    bare, hybridization, bare_renormalized, hybridization_renormalized = (
        get_downfold_matrix_extra(hamiltonian, x_matrix)
    )

    assert bare.shape == hybridization.shape == (3, 3)
    complete = get_downfold_matrix(hamiltonian, x_matrix)
    np.testing.assert_allclose(
        bare_renormalized + hybridization_renormalized, complete, atol=1e-12
    )

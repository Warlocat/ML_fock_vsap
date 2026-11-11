# ML-Fock with vSAP features

This repository is the reproducibility release for **“Transferable Machine Learning of Electronic Hamiltonians with Superposition-of-Atomic-Potentials Features”** by Chaoqun Zhang, Christian Venturella, Enzhi Chen, and Tianyu Zhu [doi: https://doi.org/10.1021/acs.jctc.6c01477]. It contains the molecular SAIAO-vSAP data pipeline, the two-block graph neural network used in production, the reported training objective, exact QM9 learning-curve split identifiers, and representative benzene-dimer geometries and transfer-integral analysis.

The code predicts a correction to a low-cost vSAP Hamiltonian in a symmetry-adapted intrinsic atomic-orbital basis. The retained GNN learns core–core and valence–valence matrix blocks separately. The resulting Fock matrix can be diagonalized for orbital energies and density-derived observables or transformed back to the AO basis for transfer-integral calculations.

## Software design

ML-Fock is organized as a layered molecular-data and learning pipeline. `mlvk.data` reads PySCF checkpoints into lightweight molecular containers; `mlvk.lo` and `mlvk.lib` construct the SAIAO-vSAP basis and physics-informed matrix features; and `GNN_Model` applies fitted preprocessing, converts orbital diagonal/off-diagonal matrix elements into PyTorch Geometric nodes and edges, and owns the configuration and trained state needed for inference. The production network contains separate message-passing modules for the core–core and valence–valence blocks, while `VKLoss` combines matrix-element error with optional orbital-energy, density-matrix, and dipole penalties. Higher-level modules provide checkpoint generation, large-basis downfolding, validation, and transfer-integral analysis. Configuration is kept in JSON, and a serialized `GNN_Model` wrapper packages the feature definitions, scaler, architecture metadata, and model weights so the same transformation is used during training and prediction.

```text
XYZ / PySCF checkpoint
        ↓
SAIAO-vSAP basis and matrix features
        ↓
feature scaling and orbital graph construction
        ↓
core-core GNN + valence-valence GNN
        ↓
predicted Fock correction → Fock matrix → orbitals and observables
```

## Capabilities

- Generate molecular PySCF checkpoints from XYZ geometries and load existing calculations.
- Construct SAIAO-vSAP node and edge features, including custom core/valence MINAO definitions.
- Train and evaluate the production two-block message-passing GNN in single or double precision.
- Predict full Fock-matrix corrections and recover orbital energies, density matrices, charges, and dipole observables.
- Downfold cc-pVTZ Hamiltonians into a MINAO-size active space and reconstruct parent-space orbitals for validation.
- Evaluate conventional 2x2 and gauge-invariant 4x4 intermolecular transfer integrals.
- Reproduce the reported QM9 subsets using the supplied split identifiers and configurations.

## Installation

The package requires Python 3.10 or newer. A CUDA-enabled PyTorch installation is recommended for training but is not required for preprocessing, inference, or the tests.

```bash
git clone git@github.com:Warlocat/ML_fock_vsap.git
cd ML_fock_vsap
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The molecular SAIAO feature code additionally requires libDMET 0.5, and vSAP feature generation requires a PySCF build that provides `RKS.get_vsap`. These two external components are used by the reference production environment but are not vendored here. Verify the environment before running data preparation:

```bash
python - <<'PY'
from pyscf import gto, scf
import libdmet

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
assert hasattr(scf.RKS(mol), "get_vsap")
print("vSAP and libDMET are available")
PY
python -m pytest
```


## Repository layout

```text
mlvk/
  basis/minao_cvs/             custom MINAO core and valence definitions
  data/                        PySCF checkpoint and dataset containers
  lib/                         feature extraction and utility routines
  lo/                          IAO/SAIAO/vSAP local-orbital construction
  model/
    gnn_model.py               dataset/inference wrapper
    preprocessing.py           feature scalers
    pytorch/                   production GNN, loss, and trainer
  workflow/
    generate.py                molecular DFT checkpoint generation
    ham_downfolding.py         Hamiltonian downfolding
    transfer_integrals.py      2x2/4x4 effective-Hamiltonian analysis
examples/
  qm9/                         cc-pVDZ final config and split identifiers
  qm9_downfold/                cc-pVTZ downfolding config, splits, validation
  transfer_integrals/          evaluator and representative XYZ geometries
tests/                         small, data-independent tests
```


## Data and Software Availability

The source code, model architecture, training and analysis scripts, final hyperparameter configuration, exact QM9 training-set identifiers, custom basis data, and representative molecular geometries underlying this study are available in this repository. The complete QM9 dataset is available from its original public source and is not redistributed here.



## Input data

The training pipeline consumes PySCF `.chk` files. Each checkpoint must contain a converged molecular calculation with the geometry, basis, overlap, core Hamiltonian, density matrix, Fock matrix, and related quantities read by `mlvk.data.Data.load_chk`. The final example uses `cc-pVDZ`, double precision, and the `saiaovsap_crude` learning basis.

The raw/derived QM9 checkpoints are not duplicated in this repository. Download QM9 structures from their public source, generate the required PySCF checkpoints consistently, and select molecules using `examples/qm9/qm9_train_ids_<N>.txt`. These lists preserve the exact learning-curve subsets from the final runs. Paths in `examples/qm9/config.json` are intentionally portable placeholders; update `train_files`, `torch_data_root`, and `gnn_model_file` for the local system. Do not change the feature order unless the corresponding input dimensions are also regenerated.

The final node features are:

1. `fock_saiaovsap`
2. `hcore_kin_saiaovsap`
3. `dm_saiaovsap`
4. `boys_saiaovsap`

The final edge features are:

1. `fock_saiaovsap`
2. `vsap_saiaovsap`
3. `dm_saiaovsap`
4. `hcore_kin_saiaovsap`

The target is `delta_fock_saiaovsap`.

## Reproducing a QM9 training run

Edit `examples/qm9/config.json` so that `train_files` names the desired generated checkpoints. To reproduce another reported training size, populate that array from the corresponding ID list. Then run:

```bash
mpirun -n 4 python -m examples.qm9.prepare_data examples/qm9/config.json
python -m examples.qm9.train examples/qm9/config.json --save-checkpoints
```

The example preserves the final 1,000-molecule hyperparameters: scale 0.01; hidden sizes 128/256; message-passing widths 64/128; two/four message-passing updates for core/valence; 500 epochs; learning rate 5e-4; batch size 10; and extra losses beginning at epoch 200 on the valence block. The objective weights are 0.1 for MO energies, 0.05 for the density matrix, and 0.05 for the dipole.

`prepare_data.py` serializes the fitted data wrapper and processed PyTorch graphs. `train.py` appends the trained state dictionary and architecture metadata to that wrapper and also writes a `_wts.pt` file.

### Prediction with a trained model

A trained `.joblib` file contains the preprocessing state, feature specification, architecture, and model weights. Predicting from a compatible PySCF checkpoint returns an `(n_orbital, n_orbital, n_target)` tensor in the learning basis:

```python
import joblib
import numpy as np

model = joblib.load("artifacts/qm9_1000/model.joblib")
delta_fock = model.predict_vk("/path/to/molecule.chk")[:, :, 0]

basis_name = model.basis.split("_")[0]
fock_ml = model.pdata[f"fock_{basis_name}"] + delta_fock
mo_energy, mo_coeff = np.linalg.eigh(fock_ml)
print(mo_energy)
```

The input checkpoint must use the basis, functional, and stored fields expected by the trained configuration. `model.pdata` holds the features generated for the most recent prediction; `C_ao_<basis>` and `C_<basis>_ao` can be used to transform matrices between the learning and AO bases.

## Reproducing the cc-pVTZ downfolding runs

The downfolding production configuration is in `examples/qm9_downfold/config.json`. It uses a cc-pVTZ parent calculation, projects to the MINAO-size SAIAO-vSAP active space, and has 54 node and edge input channels. Its target is the exactly downfolded converged cc-pVTZ Fock matrix, `fock_true_saiaovsap_downfold_exact`. Exact split lists are supplied for 1k, 2k, 4k, 8k, 16k, and 32k molecules.

The complete exercised path is retained:

1. `mlvk.data.Data.load_chk` loads the cc-pVTZ checkpoint and requests `saiaovsap_crude_downfold` features.
2. `mlvk.lib.ml_helper.load_md_saiao_vsap` constructs and orders the SAIAO-vSAP basis.
3. `mlvk.lib.ml_helper.get_downfold_features` builds the exact target and vSAP-derived approximate transformation.
4. `mlvk.workflow.ham_downfolding.get_X_matrix`, `get_downfold_matrix`, and `get_downfold_matrix_extra` construct the renormalized active-space matrices and bare/hybridization features.
5. `mlvk.model.GNN_Model.data_to_graph` selects the downfolded overlap, density, dipoles, and target for training.
6. `mlvk.workflow.get_ml_info.validate_model_mol_downfold` lifts predicted active-space orbitals back to the parent basis and computes orbital, density, charge, and dipole validation metrics.

After updating paths in the downfolding configuration, prepare and train it with the same entry points:

```bash
mpirun -n 4 python -m examples.qm9.prepare_data examples/qm9_downfold/config.json
python -m examples.qm9.train examples/qm9_downfold/config.json --save-checkpoints
```

For MPI validation, edit `examples/qm9_downfold/validation_spec.json` and run:

```bash
mpirun -n 4 python -m examples.qm9_downfold.validate \
  examples/qm9_downfold/validation_spec.json \
  artifacts/qm9_downfold_validation.csv
```

## Transfer-integral example

The `effective_hamiltonian` routine first projects the dimer Fock and overlap matrices into fragment frontier-orbital subspaces and then applies symmetric Löwdin orthogonalization. With one frontier orbital per fragment it gives the conventional 2x2 result. With the two degenerate benzene HOMOs per fragment it produces a 4x4 effective Hamiltonian; the Frobenius norm and singular values of its intermolecular block are invariant to rotations within each degenerate subspace.

Given a trained model wrapper:

```bash
python -m examples.transfer_integrals.evaluate \
  examples/transfer_integrals/geometries/benzene_scan_4.0A_0deg.xyz \
  /path/to/model.joblib \
  --atoms-fragment1 12 \
  --functional camb3lyp \
  --basis ccpvdz \
  --carrier hole \
  --degeneracy 2
```

For a nondegenerate frontier orbital, use `--degeneracy 1`. The geometry reader assumes that all atoms of fragment 1 precede all atoms of fragment 2 in the XYZ file. The command runs the baseline calculations, predicts the three ML Fock matrices (two monomers and one dimer), and reports reference and ML values in eV.

The geometry directory contains three examples sampled from the 2,000-geometry benzene training set and three scan structures at 4.0 or 6.0 Å. They are examples, not the complete dimer dataset.



## License

GPL-3.0. See `LICENSE`.

## Use of generative AI

OpenAI Codex, powered by OpenAI GPT-5 (model version: GPT-5.6 Sol; accessed through the Codex desktop application), was used in preparing this public software release to assist with codebase review, selection and refactoring of production code, documentation, portable usage examples, and automated tests. All AI-assisted changes were reviewed, edited, and validated by the authors. The AI system was not used to generate the underlying quantum-chemistry calculations or training data, and the authors take full responsibility for the software and its scientific use.

# ML-Fock with vSAP features

This repository is the reproducibility release for **“Transferable Machine Learning of Electronic Hamiltonians with Superposition-of-Atomic-Potentials Features”** by Chaoqun Zhang, Christian Venturella, Enzhi Chen, and Tianyu Zhu ([arXiv:2606.12326](https://arxiv.org/abs/2606.12326)). It contains the molecular SAIAO-vSAP data pipeline, the two-block graph neural network used in production, the reported training objective, exact QM9 learning-curve split identifiers, and representative benzene-dimer geometries and transfer-integral analysis.

The code predicts a correction to a low-cost vSAP Hamiltonian in a symmetry-adapted intrinsic atomic-orbital basis. The retained GNN learns core–core and valence–valence matrix blocks separately. The resulting Fock matrix can be diagonalized for orbital energies and density-derived observables or transformed back to the AO basis for transfer-integral calculations.

## Scope of this release

Included:

- molecular checkpoint loading and SAIAO-vSAP feature construction;
- custom core/valence MINAO data used by the final configurations;
- exact and approximate Hamiltonian downfolding routines;
- `GNN_Multiblock_faster`, with the core–core and valence–valence networks used in the paper;
- matrix-element, MO-energy, density-matrix, and dipole loss terms;
- graph preparation, training, inference, and model serialization;
- exact cc-pVDZ QM9 subsets (1k–16k) and cc-pVTZ downfolding subsets (1k–32k) as molecule-ID lists;
- six representative benzene-dimer geometries; and
- conventional 2x2 and gauge-invariant 4x4 frontier-subspace transfer-integral analysis.

Deliberately excluded because they were not used for the reported production results:

- fragment-specific training and fragment-ID generation;
- intra/inter-fragment transfer-penalty loss terms;
- Fourier, k-point, JK-decomposed, multiscale, and older GNN variants;
- generic and fragment-only training loops;
- checkpoints, trained weights, cluster logs, caches, and absolute cluster paths;
- the full QM9-derived checkpoint dataset.

The original production JSON contained `beta_mo_coeff=0.05`, but the production loss code did not read that key. It is therefore omitted here rather than presenting an inactive option as part of the method.

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

## Installation

A GPU is recommended for training but is not needed for data preparation, transfer-integral analysis, or the small tests. The code was audited in an environment with Python 3.13.5, PySCF 2.12.1, NumPy 2.2.6, SciPy 1.16.3, PyTorch 2.9.1, and PyTorch Geometric 2.7.0. The production workflow also imports libDMET 0.5 for its IAO utilities.

```bash
git clone git@github.com:Warlocat/ML_fock_vsap.git
cd ML_fock_vsap
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install libDMET separately from the [libDMET preview repository](https://github.com/gkclab/libdmet_preview), then verify that `import libdmet` succeeds. CUDA-enabled PyTorch should be installed using the command appropriate for the local CUDA driver before `pip install -e .` if GPU training is desired.

The static cc-pVTZ downfolding workflow used for the paper requires only the dependencies above. The retained dynamical Green's-function/analytic-continuation utilities in `ham_downfolding.py` additionally import [fcdmft](https://github.com/Warlocat/fcdmft); the audited environment used commit `9bf57ab64d270fd3ee6358430e22c77c598d0e0c`. Install that project only when using `solve_Heff_AC`, `get_gf_fromAC`, or the Padé/grid helper routines.

Run the data-independent checks with:

```bash
python -m pytest
```

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

## Reproducibility boundaries

This repository is intended to make the implemented method and the reported configuration inspectable and runnable. Reproducing numerical tables exactly additionally requires the original derived PySCF checkpoints and trained model weights. Before the JCTC revision is finalized, those artifacts should either be deposited in Zenodo/figshare or supplied as Supporting Information. At minimum, deposit:

- trained model wrappers or state dictionaries used for every reported figure/table;
- all non-QM9 dimer geometries used in training, validation, and scans;
- machine-readable reference and ML values behind each plot;
- the scripts/configurations mapping deposited inputs to each figure and table;
- package/environment lock information and the exact code commit; and
- a manifest with checksums, units, basis, functional, charge/spin, and split membership.

A GitHub repository alone is mutable. For the manuscript, archive a tagged release in Zenodo and cite the resulting DOI.

## Data and Software Availability

The source code, model architecture, training and analysis scripts, final hyperparameter configuration, exact QM9 training-set identifiers, custom basis data, and representative molecular geometries underlying this study are available in this repository. The complete QM9 dataset is available from its original public source and is not redistributed here. Derived data and trained model parameters needed to reproduce the reported numerical results are available at **[insert archival repository DOI]**. The version corresponding to the manuscript is archived as **[insert release tag/DOI]** and is distributed under the GNU General Public License v3.0.

Replace the bracketed archival references before submission.

## Citation

Please cite the manuscript:

```bibtex
@article{zhang2026transferable,
  title   = {Transferable Machine Learning of Electronic Hamiltonians with Superposition-of-Atomic-Potentials Features},
  author  = {Zhang, Chaoqun and Venturella, Christian and Chen, Enzhi and Zhu, Tianyu},
  year    = {2026},
  eprint  = {2606.12326},
  archivePrefix = {arXiv},
  primaryClass = {physics.chem-ph},
  doi     = {10.48550/arXiv.2606.12326}
}
```

## License

GPL-3.0. See `LICENSE`.

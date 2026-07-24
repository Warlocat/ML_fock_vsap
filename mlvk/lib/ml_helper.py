from functools import reduce
import numpy as np
import joblib
import warnings
import pandas as pd
import scipy
from scipy.spatial import distance_matrix

import pyscf
from pyscf import lib, dft, lo
from pyscf.lib import logger

from mlvk.lo.saao import get_C_ao_iao, get_saao, get_custom_minao_dat, lowdin_orthk, get_saaok
from mlvk.lo.saiao_vsap import get_C_ao_iao_vsap, reorder_iaofirst, reorder_ao_labels_iaofirst
from mlvk.lib.utils import plot_performance

def get_sapbasis(mol, sap_basis):
    from pyscf.gto.basis import load
    if isinstance(sap_basis, str):
        atoms = [coord[0] for coord in mol._atom]
        sapbas = {}
        for atom in set(atoms):
            single_element_bs = load(sap_basis, atom)
            if isinstance(single_element_bs, dict):
                sapbas[atom] = np.asarray(single_element_bs[atom][0][1:], dtype=float)
            else:
                sapbas[atom] = np.asarray(single_element_bs[0][1:], dtype=float)
    elif isinstance(sap_basis, dict):
        sapbas = {}
        for key in sap_basis:
            sapbas[key] = np.asarray(sap_basis[key][0][1:], dtype=float)
    
    else:
        raise ValueError('invalid sap_basis')

    return sapbas

def get_vsap_features(mf, s, hcore, sap_type='dft', sap_basis = 'sapgrasplarge'):
    from pyscf.scf.hf import make_sap
    mol = mf.mol
    assert sap_type.lower() in ['dft', 'hf']
    
    if sap_type.lower() == 'dft':
        # DFT get_vsap returns Vsap = Vnuc + J + Vxc(LDA)
        Vsap = mf.get_vsap()
        Vsap -= mol.intor("int1e_nuc")  # remove nuclear part
    else:
        # HF make_sap returns effective J
        # Vsap = Vnuc + J
        Vsap = make_sap(mol, sap_basis=get_sapbasis(mol, sap_basis))
    e, coeff = scipy.linalg.eigh(hcore + Vsap, s) # IDEA: can make green's function from this?

    mf = pyscf.scf.hf.RHF(mol)
    occ = mf.get_occ(e, coeff)

    dm = mf.make_rdm1(coeff, occ)
    return dm, Vsap, e, coeff

def get_atmf_features(mf, s, hcore):
    from pyscf.scf.hf import make_sap
    mol = mf.mol
    if len(mol._ecpbas) > 0:
        raise NotImplementedError("atm features not implemented for ECP calculations")
    mol._ecpbas = np.zeros((0,8), dtype=np.int32)

    from pyscf.dft.rks import RKS
    from pyscf.scf.hf import init_guess_by_atom
    dm = init_guess_by_atom(mol)
    mf_tmp = RKS(mol, xc = "lda,vwn")
    mf_tmp.grids.level = 1
    fock = mf_tmp.get_veff(dm=dm) + hcore
    e, coeff = scipy.linalg.eigh(fock, s)
    occ = mf_tmp.get_occ(e, coeff)
    dm = mf_tmp.make_rdm1(coeff, occ)
    return dm, fock, e, coeff

def load_md_saiao(mlf, fname, force_eigv_direction = True, custom_minao_path = None, mol = None, mf = None, blockdiag_on = 'fock'):

    if mol is None:
        mol = pyscf.lib.chkfile.load_mol(fname)
    
    if mf is None:
        mf = pyscf.dft.RKS(mol)
        mf.xc = mlf.get('xc', 'hf')
        if type(mf.xc) is bytes:
            mf.xc = mf.xc.decode('utf-8')
        scf_data = pyscf.lib.chkfile.load(fname, 'scf')
        mf.__dict__.update(scf_data)

    C_ao_iao, C_iao_saiao, fock_iao = get_chk_saiao(mf, mlf[blockdiag_on], minao = 'minao', force_eigv_direction = force_eigv_direction, val_core_dats = get_custom_minao_dat(mol,custom_minao_path))
    C_ao_saiao = np.dot(C_ao_iao, C_iao_saiao)
    mlf = get_lo_features(mol, mlf, C_ao_saiao, basis_name = 'saiao')

    mlf['fock_iao'] = fock_iao
    mlf['C_ao_iao'] = C_ao_iao
    mlf['C_iao_saiao'] = C_iao_saiao
    
    return mlf

def load_md_saiao_vsap(mlf, fname, force_eigv_direction = True, custom_minao_path = None, mol = None, mf = None, blockdiag_on = 'fock', sap_type = 'dft', reorder_saiao = True, downfold = False, downfold_basis="minao"):

    if downfold_basis is None or (not downfold):
        downfold_basis = 'minao'

    if mol is None:
        mol = pyscf.lib.chkfile.load_mol(fname)
    
    if mf is None:
        mf = pyscf.dft.RKS(mol)
        mf.xc = mlf.get('xc', 'hf')
        if type(mf.xc) is bytes:
            mf.xc = mf.xc.decode('utf-8')
        scf_data = pyscf.lib.chkfile.load(fname, 'scf')
        mf.__dict__.update(scf_data)

    if blockdiag_on == 'hcore+vsap':
        hv = mf.get_vsap() + mol.intor("int1e_kin")
    else:
        hv = mlf[blockdiag_on]
    val_core_dats = get_custom_minao_dat(mol,custom_minao_path)
    C_ao_iao, C_iao_saiao, fock_iao = get_chk_saiao_vsap(mf, hv, minao = downfold_basis, force_eigv_direction = force_eigv_direction, val_core_dats = val_core_dats, sap_type=sap_type, reorder_saiao = reorder_saiao)
    C_ao_saiao = np.dot(C_ao_iao, C_iao_saiao)

    if reorder_saiao:
        if val_core_dats is None:
            ao_labels = reorder_ao_labels_iaofirst(mol, minao=downfold_basis)
        else:
            ao_labels = reorder_ao_labels_iaofirst(mol, minao=downfold_basis, minao_val=val_core_dats[0], minao_core=val_core_dats[1])
    else:
        ao_labels = mol.ao_labels()

    mlf = get_lo_features(mol, mlf, C_ao_saiao, basis_name = 'saiaovsap', ao_labels = ao_labels)

    mlf['fock_iao_vsap'] = fock_iao
    mlf['C_ao_iao_vsap'] = C_ao_iao
    mlf['C_iao_saiaovsap'] = C_iao_saiao

    if downfold:
        assert reorder_saiao, "downfolding currently only implemented for reordering to iao first, pao second"
        from pyscf.lo.iao import reference_mol
        pmol = reference_mol(mol, minao=downfold_basis)
        nB2 = len(pmol.ao_labels())
        pmol_ao_labels = ao_labels[:nB2]
        mlf = get_downfold_features(mol, mlf, minao = downfold_basis, basis_name = 'saiaovsap', ao_labels=pmol_ao_labels)
    
    return mlf

def load_md_saiao_atmf(mlf, fname, force_eigv_direction = True, custom_minao_path = None, mol = None, mf = None, blockdiag_on = 'hcore', sap_type = 'dft', reorder_saiao = True, downfold = False, downfold_basis="minao"):

    if downfold_basis is None or (not downfold):
        downfold_basis = 'minao'

    if mol is None:
        mol = pyscf.lib.chkfile.load_mol(fname)
    
    if mf is None:
        mf = pyscf.dft.RKS(mol)
        mf.xc = mlf.get('xc', 'hf')
        if type(mf.xc) is bytes:
            mf.xc = mf.xc.decode('utf-8')
        scf_data = pyscf.lib.chkfile.load(fname, 'scf')
        mf.__dict__.update(scf_data)

    if blockdiag_on == 'hcore+vsap':
        hv = mf.get_vsap() + mol.intor("int1e_kin")
    else:
        hv = mlf[blockdiag_on]
    val_core_dats = get_custom_minao_dat(mol,custom_minao_path)
    C_ao_iao, C_iao_saiao, fock_iao = get_chk_saiao_vsap(mf, hv, minao = downfold_basis, force_eigv_direction = force_eigv_direction, val_core_dats = val_core_dats, sap_type=sap_type, reorder_saiao = reorder_saiao)
    C_ao_saiao = np.dot(C_ao_iao, C_iao_saiao)

    if reorder_saiao:
        if val_core_dats is None:
            ao_labels = reorder_ao_labels_iaofirst(mol, minao=downfold_basis)
        else:
            ao_labels = reorder_ao_labels_iaofirst(mol, minao=downfold_basis, minao_val=val_core_dats[0], minao_core=val_core_dats[1])
    else:
        ao_labels = mol.ao_labels()

    mlf = get_lo_features(mol, mlf, C_ao_saiao, basis_name = 'saiaovsap', ao_labels = ao_labels)

    mlf['fock_iao_vsap'] = fock_iao
    mlf['C_ao_iao_vsap'] = C_ao_iao
    mlf['C_iao_saiaovsap'] = C_iao_saiao

    if downfold:
        assert reorder_saiao, "downfolding currently only implemented for reordering to iao first, pao second"
        from pyscf.lo.iao import reference_mol
        pmol = reference_mol(mol, minao=downfold_basis)
        nB2 = len(pmol.ao_labels())
        pmol_ao_labels = ao_labels[:nB2]
        mlf = get_downfold_features(mol, mlf, minao = downfold_basis, basis_name = 'saiaovsap', ao_labels=pmol_ao_labels)
    
    return mlf

def load_md_lao(mlf, fname, force_eigv_direction = True, mol = None, orth_method = 'meta_lowdin'):

    if mol is None:
        mol = pyscf.lib.chkfile.load_mol(fname)

    C_ao_lao = lo.orth_ao(mol, method = orth_method)
    # hcore_lao = reduce(np.dot, (C_ao_lao.T, mlf['hcore'], C_ao_lao))
    # C_ao_lao, angular_idx = get_saao(mol, hcore_lao, force_eigv_direction = force_eigv_direction, return_idx = True)
    # C_ao_salao = np.dot(C_ao_lao, C_lao_salao)
    
    mlf = get_lo_features(mol, mlf, C_ao_lao, basis_name = 'lao')

    # mlf['hcore_lao'] = hcore_lao
    # mlf['C_ao_lao'] = C_ao_lao
    # mlf['C_lao_lao'] = C_lao_lao
    
    return mlf

def load_md_salao(mlf, fname, force_eigv_direction = True, mol = None, orth_method = 'meta_lowdin', blockdiag_on = 'hcore'):

    if mol is None:
        mol = pyscf.lib.chkfile.load_mol(fname)

    C_ao_lao = lo.orth_ao(mol, method = orth_method)
    hcore_lao = reduce(np.dot, (C_ao_lao.T, mlf[blockdiag_on], C_ao_lao))
    C_lao_salao, angular_idx = get_saao(mol, hcore_lao, force_eigv_direction = force_eigv_direction, return_idx = True)
    C_ao_salao = np.dot(C_ao_lao, C_lao_salao)
    
    mlf = get_lo_features(mol, mlf, C_ao_salao, basis_name = 'salao')

    mlf['hcore_lao'] = hcore_lao
    mlf['C_ao_lao'] = C_ao_lao
    mlf['C_lao_salao'] = C_lao_salao
    
    return mlf


def hermitize_feature(f):
    norbs = f.shape[-1]
    iu = np.array(np.triu_indices(norbs, 1)) # for the assignment of full vk
    f[:,iu[1,:], iu[0,:]] = f[:,iu[0,:], iu[1,:]].conj()
    return f


def load_md_salaok(mlf, fname, force_eigv_direction = True, cell = None, orth_method = 'meta_lowdin', hermitize_keys = ['fock', 'hcore', 'vj', 'vk', 'vxc', 'dm_hf', 'ovlp']):

    for key in hermitize_keys:
        if key in mlf.keys():
            mlf[key] = hermitize_feature(mlf[key])
        if f'{key}_true' in mlf.keys():
            mlf[f'{key}_true'] = hermitize_feature(mlf[f'{key}_true'])

    if cell is None:
        cell = pyscf.pbc.lib.chkfile.load_cell(fname)

    mf = pyscf.pbc.scf.khf.KSCF(cell, mlf['kpts'])
    mf.xc = mlf.get('xc', 'hf')
        # if type(mf.xc) is bytes:
        #     mf.xc = mf.xc.decode('utf-8')
        # scf_data = pyscf.lib.chkfile.load(fname, 'scf')
        # mf.__dict__.update(scf_data)

    C_ao_lao = lowdin_orthk(mf, method = orth_method, s = mlf['ovlp'])
    hcore_lao = rotate_rank3(mlf['hcore'], C_ao_lao)
    hcore_lao = hermitize_feature(hcore_lao)
    C_lao_salao = get_saaok(cell, hcore_lao, force_eigv_direction = force_eigv_direction)
    C_ao_salao = np.asarray([np.dot(C_ao_lao[i], C_lao_salao[i]) for i in range(C_ao_lao.shape[0])])
    mlf = get_lok_features(cell, mlf, C_ao_salao, basis_name = 'salaok')

    for key in hermitize_keys:
        if f'{key}_salaok' in mlf.keys():
            mlf[f'{key}_salaok'] = hermitize_feature(mlf[key])
        if f'{key}_true_salaok' in mlf.keys():
            mlf[f'{key}_true_salaok'] = hermitize_feature(mlf[f'{key}_true_salaok'])

    mlf['hcore_lao'] = hcore_lao
    mlf['C_ao_lao'] = C_ao_lao
    mlf['C_lao_salao'] = C_lao_salao
    
    return mlf

# def load_md_salao(mlf, fname, force_eigv_direction = True, mol = None, orth_method = 'meta_lowdin'):

#     if mol is None:
#         mol = pyscf.lib.chkfile.load_mol(fname)

#     ovlp = mlf['ovlp']
#     hcore = mlf['ovlp']
#     C_ao_saao = get_saao(mol, ovlp, force_eigv_direction = True)
#     S_saao = np.linalg.multi_dot((C_ao_saao.T, ovlp, C_ao_saao)) 
#     C_saao_salao = lowdin_orth_wrapper(C_ao_saao, S_saao)
#     C_ao_salao = np.dot(C_ao_saao, C_saao_salao)
#     # hcore_salao = np.linalg.multi_dot((C_ao_salao.T, hcore, C_ao_salao)) 
#     # C_salao_shalao = get_saao(mol, hcore_salao, force_eigv_direction = True)
#     # C_ao_shalao = np.dot(C_ao_salao, C_salao_shalao)
    
#     mlf = get_lo_features(mol, mlf, C_ao_salao, basis_name = 'salao')

#     mlf['S_saao'] = S_saao
#     mlf['C_ao_saao'] = C_ao_saao
#     mlf['C_saao_salao'] = C_saao_salao
#     # mlf['C_salao_shalao'] = C_salao_shalao
    
#     return mlf

def load_md_saao(mlf, fname, force_eigv_direction = True, mol = None):
    if mol is None:
        mol = pyscf.lib.chkfile.load_mol(fname)

    mlf['inds_core'] = get_core_orbital_indices(mol.ao_labels())
    C_ao_saao, angular_idx = get_saao(mol, mlf['hcore'], force_eigv_direction = force_eigv_direction, return_idx = True)
    mlf = get_lo_features(mol, mlf, C_ao_saao, basis_name = 'saao')
    mlf['angular_idx'] = angular_idx

    return mlf

def convert_to_newbasis(mol, mlf, newbasis):

    assert newbasis in ['saao', 'saiao', 'lao']

    mlf['nocc'] = mol.nelectron // 2
    mf = pyscf.dft.RKS(mol)
    mf.xc = mlf.get('xc', 'hf')
    if type(mf.xc) is bytes:
        mf.xc = mf.xc.decode('utf-8')

    mf.mo_energy, mf.mo_coeff = mf.eig(mlf['fock'], mlf['ovlp'])
    
    mlf['mo_energy'] = mf.mo_energy
    mlf['mo_coeff'] = mf.mo_coeff
    mf.mo_occ = mf.get_occ(mlf['mo_energy'], mlf['mo_coeff'])
    mlf['mo_occ'] = mf.mo_occ
    mlf['dm_hf'] = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    if newbasis == 'saao':
        return load_md_saao(mlf, None, mol = mol) # probably best to keep this the same hcore for boosting (not vj and vk)
    elif newbasis == 'saiao':
        return load_md_saiao(mlf, None, mol = mol, mf = mf, custom_minao_path = mlf.get('custom_minao_path', None)) # probably best to keep this the same hcore for boosting (not vj and vk)
    elif newbasis == 'lao':
        return load_md_lao(mlf, None, mol = mol) # probably best to keep this the same hcore for boosting (not vj and vk)

def extract_ao_features(mlf):
    keys = ['hcore', 'ovlp', 'mo_coeff', 'mo_energy', 'nocc', 'dm_hf', 'fock', 'vj', 'vxc', 'vk', 'xc', 
        'C_saao_ao', 'C_ao_saao', 'C_ao_saiao', 'C_saiao_ao', 'C_ao_lao', 'C_lao_ao', 'hcore+vj']
    new_mlf = {}
    for key in keys:
        if key in mlf.keys():
            try:
                new_mlf[key] = mlf[key].copy()
            except AttributeError:
                new_mlf[key] = mlf[key]

    return new_mlf

# def terminate_chained_prediction(mlf, veff_lo, C_ao_lo, target_names):

#     for i in range(veff_lo.shape[-1]):
#         key = target_names[i]
#         veff_lo[:,:,i] = np.linalg.multi_dot((C_ao_lo.T, mlf[key], C_ao_lo))
    
#     return veff_lo
            
def get_features_delta(mlf, mol, veff_lo, target_names, oldbasis, newbasis, file_ref = None, delta = False):
    """create features for boosted delta learning of vk between two different representations

    Args:
        mlf: dict-like object
        mol: pyscf mol object for basis rotation
        veff_lo: the machine-learned veff, naoxnaoxm, where m = 3 for three vk, vj, vxc prediction
        target_names: list of names like vxc, vj, vk to assign the name identity of ML features
        old_basis: basis that ML prediction is in
        new_basis: basis to create from ML prediction (repolarized the mean-field wrt to ML output)
        file_ref: basis of labels for ML if supplied
    Returns:
        mlf: dict-like
    """    

    assert newbasis in ['saao', 'saiao', 'lao']
    C_lo_ao = mlf[f'C_{oldbasis}_ao'].copy()
    mlf['fock'] = np.zeros_like(C_lo_ao)
    for i in range(veff_lo.shape[-1]):
        
        key = target_names[i]
        if delta:
            mlf[key] += np.linalg.multi_dot((C_lo_ao.T, veff_lo[:,:,i], C_lo_ao)) # in ao
        else:
            mlf[key] = np.linalg.multi_dot((C_lo_ao.T, veff_lo[:,:,i], C_lo_ao)) # in ao
        
        mlf['fock'] += mlf[key].copy()
        
    
    # create the new feature from predicted vj, vk, vxc
    if 'vj' not in mlf.keys() and 'hcore+vj' in mlf.keys():
        mlf['vj'] = mlf['hcore+vj'] - mlf['hcore']
        

    mlf = convert_to_newbasis(mol, mlf, newbasis)
    
    # collect the targets (deltas between true and current)
    if file_ref is not None:
        
        C_ao_lo = mlf[f'C_ao_{newbasis}'] # take care, this rotation should not come from true target quantities, and be used to rotate true and predicted
        mlf_true = lib.chkfile.load(file_ref, 'mlf')
        mlf_true['hcore+vj'] = mlf_true['hcore'] +  mlf_true['vj']
        
        # this is in true target in ML-derived IAO basis
        for key in target_names:
            # plot_performance(mlf[key], mlf_true[key], name = key)
            mlf[f'target_{key}_{newbasis}'] = np.linalg.multi_dot((C_ao_lo.T, mlf_true[key].copy(), C_ao_lo))
            mlf[f'delta_{key}_{newbasis}'] = np.linalg.multi_dot((C_ao_lo.T,  mlf_true[key].copy() - mlf[key].copy(), C_ao_lo))

        # for loss fn
        mlf[f'fock_true_{newbasis}'] = np.linalg.multi_dot((C_ao_lo.T, mlf_true['fock'], C_ao_lo))
        mlf['fname'] = file_ref

    return mlf


def get_features_delta_samebasis(mlf, mol, veff_lo, target_names, basis, file_ref = None, delta = False):
    """create features for boosted delta learning of vk between two different representations

    Args:
        mlf: dict-like object
        mol: pyscf mol object for basis rotation
        veff_lo: the machine-learned veff, naoxnaoxm, where m = 3 for three vk, vj, vxc prediction
        target_names: list of names like vxc, vj, vk to assign the name identity of ML features
        basis: basis that ML prediction is in
        file_ref: basis of labels for ML if supplied
    Returns:
        mlf: dict-like
    """    
    basis = basis.split('_')[0]
    C_lo_ao = mlf[f'C_{basis}_ao'].copy()
    C_ao_lo = mlf[f'C_ao_{basis}'] # take care, this rotation should not come from true target quantities, and be used to rotate true and predicted
    mlf['fock'] = np.zeros_like(C_lo_ao)
    for i in range(veff_lo.shape[-1]):
        
        key = target_names[i]
        if delta:
            mlf[key] += np.linalg.multi_dot((C_lo_ao.T, veff_lo[:,:,i], C_lo_ao)) # in ao
        else:
            mlf[key] = np.linalg.multi_dot((C_lo_ao.T, veff_lo[:,:,i], C_lo_ao)) # in ao
        
        mlf['fock'] += mlf[key].copy()
        
    
    # create the new feature from predicted vj, vk, vxc
    if 'vj' not in mlf.keys() and 'hcore+vj' in mlf.keys():
        mlf['vj'] = mlf['hcore+vj'] - mlf['hcore']

    mlf = get_lo_features(mol, mlf, C_ao_lo, basis_name = basis, do_assertions = False)

    # collect the targets (deltas between true and current)
    if file_ref is not None:
        
        
        mlf_true = lib.chkfile.load(file_ref, 'mlf')
        mlf_true['hcore+vj'] = mlf_true['hcore'] +  mlf_true['vj']
        
        # this is in true target in ML-derived IAO basis
        for key in target_names:
            # plot_performance(mlf[key], mlf_true[key], name = key)
            mlf[f'target_{key}_{basis}'] = np.linalg.multi_dot((C_ao_lo.T, mlf_true[key].copy(), C_ao_lo))
            mlf[f'delta_{key}_{basis}'] = mlf[f'target_{key}_{basis}'] -  mlf[f'{key}_{basis}'].copy()

        # for loss fn
        mlf[f'fock_true_{basis}'] = np.linalg.multi_dot((C_ao_lo.T, mlf_true['fock'], C_ao_lo))
        mlf['fname'] = file_ref

    return mlf

def get_mo_features(mlf):
    """get ML features in MO basis, not really used.

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py

    Returns:
        dict: modified mlf dictionary with MO basis features
    """    

    basis_name = 'mo'
    mo_energy = mlf['mo_energy']
    mo_coeff = mlf['mo_coeff']
    nocc = mlf['nocc']
    mo_occ = mlf['mo_occ']
            
    # feature 1: fock matrix
    fock_mo = np.diag(mo_energy)

    # feature 2 : density matrix
    dm_mo = np.diag(mo_occ)

    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_mo = np.linalg.multi_dot((mo_coeff.T, hcore, mo_coeff))

    # features 4 & 5 : J and K matrices
    vj_mo = np.linalg.multi_dot((mo_coeff.T,  mlf['vj'], mo_coeff))

    mlf[f'dm_{basis_name}'] = dm_mo
    mlf[f'fock_{basis_name}'] = fock_mo
    mlf[f'hcore_{basis_name}'] = hcore_mo
    mlf[f'vj_{basis_name}'] = vj_mo

    if 'vxc' in mlf.keys():
        vxc_mo = np.linalg.multi_dot((mo_coeff.T, mlf['vxc'], mo_coeff))
        mlf[f'vxc_{basis_name}'] = vxc_mo
    
    if 'vk' in mlf.keys():
        vk_mo = np.linalg.multi_dot((mo_coeff.T, mlf['vk'], mo_coeff))
        mlf[f'vk_{basis_name}'] = vk_mo
    
    return mlf

def get_lo_features(mol, mlf, C_ao_lo, basis_name = 'saao', ao_labels = None, do_assertions = False):
    """get ML features in lo basis

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py
        C_ao_lo (np.float64, norb x norb): rotation matrix from AO to LO

    Returns:
        dict: modified mlf dictionary with lo basis features
    """
    if ao_labels is None:
        ao_labels = mol.ao_labels()

    mlf[f'C_ao_{basis_name}'] = C_ao_lo

    # mo_energy = mlf['mo_energy']
    # mo_coeff = mlf['mo_coeff']
    S_ao = mlf['ovlp']
    nocc = mlf['nocc']
    nmo = S_ao.shape[0] #nao =  mol.nao_nr() #len(mo_energy)#
    dm = mlf['dm_hf']

    mlf['hcore_nuc'] = mol.intor_symmetric('int1e_nuc') 
    mlf['hcore_kin'] = mol.intor_symmetric('int1e_kin')
    dip_ints = mol.intor_symmetric('int1e_r', comp=3)
    mlf['dipole_x'] = dip_ints[0]
    mlf['dipole_y'] = dip_ints[1]
    mlf['dipole_z'] = dip_ints[2]

    nelectron = nocc*2
    S_lo = np.linalg.multi_dot((C_ao_lo.T, S_ao, C_ao_lo))
            
    # feature 1: fock matrix
    fock = mlf['fock']
    fock_lo = np.linalg.multi_dot((C_ao_lo.T, fock, C_ao_lo))
    

    S_lo_inv = np.linalg.inv(S_lo)
    SCS = np.linalg.multi_dot((S_lo_inv, C_ao_lo.T, S_ao))
    dm_lo = np.linalg.multi_dot((SCS, dm, SCS.T))
   
    # print(diff_mo)
    if do_assertions:
        mo_energy = mlf['mo_energy']
        mo_energy_new, mo_coeff_new = scipy.linalg.eigh(fock_lo, S_lo)
        diff_mo = np.max(np.abs(mo_energy_new - mo_energy))
        fock_ao_check = np.linalg.multi_dot((SCS.T, fock_lo, SCS))
        assert ((diff_mo)< 1e-4)
        # transform Fock from lo back to AO

        assert ((np.max(np.abs(fock_ao_check-fock)))<1e-6)

        # feature 2 : density matrix
        assert(abs(np.trace(np.dot(dm_lo, S_lo))-mol.nelectron)<1e-8)

        # transform dm from lo back to AO
        dm_ao_new = np.linalg.multi_dot((C_ao_lo, dm_lo, C_ao_lo.T))
        assert ((np.max(np.abs(dm_ao_new-dm)))<1e-6)

    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_lo = np.linalg.multi_dot((C_ao_lo.T, hcore, C_ao_lo))
    
    # MO to lo rotations for hamiltonian-like matrices
    # C_lo_mo = np.linalg.multi_dot((S_lo_inv, C_ao_lo.T, S_ao, mo_coeff))
    # C_mo_lo = np.linalg.multi_dot((mo_coeff.T, S_ao, C_ao_lo, S_lo_inv))
    # C_mo_lo = np.linalg.multi_dot((mo_coeff.T, S_ao, C_ao_lo))
    
    mlf[f'dm_{basis_name}'] = dm_lo
    mlf[f'fock_{basis_name}'] = fock_lo
    mlf[f'hcore_{basis_name}'] = hcore_lo
    if 'mo_coeff' in mlf.keys():
        mlf[f'C_{basis_name}_mo'] = np.linalg.multi_dot((SCS, mlf['mo_coeff']))
    if 'mo_energy' in mlf.keys():
        mlf['ef'] = 0.5*(mlf['mo_energy'][nocc] + mlf['mo_energy'][nocc-1])
    mlf[f'C_{basis_name}_ao'] = SCS # for fock matrix
    mlf[f'S_{basis_name}'] = S_lo

    mlf[f'hcore_nuc_{basis_name}'] = np.linalg.multi_dot((C_ao_lo.T, mlf['hcore_nuc'], C_ao_lo))
    mlf[f'hcore_kin_{basis_name}'] = np.linalg.multi_dot((C_ao_lo.T, mlf['hcore_kin'], C_ao_lo))
    mlf[f'dipole_x_{basis_name}'] = np.linalg.multi_dot((C_ao_lo.T, mlf['dipole_x'], C_ao_lo))
    mlf[f'dipole_y_{basis_name}'] = np.linalg.multi_dot((C_ao_lo.T, mlf['dipole_y'], C_ao_lo))
    mlf[f'dipole_z_{basis_name}'] = np.linalg.multi_dot((C_ao_lo.T, mlf['dipole_z'], C_ao_lo))

    if 'vsap' in mlf.keys():
        vsap_lo = np.linalg.multi_dot((C_ao_lo.T,  mlf['vsap'], C_ao_lo))
        mlf[f'vsap_{basis_name}'] = vsap_lo
        mlf[f'focksap_{basis_name}'] = mlf[f'hcore_{basis_name}'] + mlf[f'vsap_{basis_name}']

    if 'vj' in mlf.keys():
        vj_lo = np.linalg.multi_dot((C_ao_lo.T,  mlf['vj'], C_ao_lo))
        mlf[f'vj_{basis_name}'] = vj_lo

    if 'vxc' in mlf.keys():
        vxc_lo = np.linalg.multi_dot((C_ao_lo.T, mlf['vxc'], C_ao_lo))
        mlf[f'vxc_{basis_name}'] = vxc_lo

    if 'vk' in mlf.keys():
        vk_lo = np.linalg.multi_dot((C_ao_lo.T, mlf['vk'], C_ao_lo))
        mlf[f'vk_{basis_name}'] = vk_lo

    true_keys = ['fock_true', 'vk_true', 'vj_true', 'vxc_true']
    for key in true_keys:
        if key in mlf.keys():
            mlf[f'{key}_{basis_name}'] = np.linalg.multi_dot((C_ao_lo.T, mlf[key], C_ao_lo))

    if 'dm_hf_true' in mlf.keys():
        mlf[f'dm_hf_true_{basis_name}'] = np.linalg.multi_dot((SCS, mlf['dm_hf_true'], SCS.T))

    if f'vj_{basis_name}' in mlf.keys():
        mlf[f'hcore+vj_{basis_name}'] = mlf[f'hcore_{basis_name}'] + mlf[f'vj_{basis_name}']
        mlf[f'vj_{basis_name}/nelec'] = mlf[f'vj_{basis_name}'] / mol.nelectron    


    if f'vj_true_{basis_name}' in mlf.keys():
        mlf[f'hcore+vj_true_{basis_name}'] = mlf[f'hcore_{basis_name}'] + mlf[f'vj_true_{basis_name}']
        mlf[f'vj_true_{basis_name}/nelec'] = mlf[f'vj_true_{basis_name}'] / mol.nelectron   

    mlf[f'hcore_{basis_name}/nelec'] = mlf[f'hcore_{basis_name}'] / mol.nelectron

    if f'fock_true_{basis_name}' in mlf.keys() and f'focksap_{basis_name}' in mlf.keys():
        mlf[f'delta_fock_{basis_name}'] = mlf[f'fock_true_{basis_name}'] - mlf[f'fock_{basis_name}']

    # extra simple orbital features

    df_mol = get_orbtypes_df(mol, ao_labels)
    mlf['inds_core'] = get_core_orbital_indices(ao_labels)
    mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = np.diag(df_mol['principal']), np.diag(df_mol['angular'])
    mlf[f'atomic_charge_{basis_name}'] = get_saiao_charges(df_mol, mlf[f'dm_{basis_name}'])
    mlf[f'boys_{basis_name}'] = get_saiao_locality(mol, mlf[f'C_ao_{basis_name}'])

    charges = np.array(df_mol['nuclear_charge'])
    mlf['nuclear_charge'] = np.abs(charges[:, None] - charges[None, :])
    np.fill_diagonal(mlf['nuclear_charge'], charges)
    coordinates = np.array(df_mol[['coordinate_x', 'coordinate_y', 'coordinate_z']])
    mlf['atomic_distance'] = distance_matrix(coordinates, coordinates)
    # mlf['1/r'] = (1.0/mlf['atomic_distance']).fillna(0.5)
    # mlf['1/r^2'] = mlf['1/r']**2
    mlf['n_atoms'] = len(mol._atom)
    mlf['atomic_index'] = list(df_mol['atm_id'])

    return mlf

def rotate_rank3(mat, rot, ovlp = None):
    if ovlp is None:
        return np.array([rot[i].conj().T @ mat[i] @ rot[i] for i in range(rot.shape[0])])
    else:
        return np.array([rot[i].conj().T @ ovlp[i] @ mat[i] @ ovlp[i] @ rot[i] for i in range(rot.shape[0])])

def get_lok_features(cell, mlf, C_ao_lo, basis_name = 'salaok', ao_labels = None, do_assertions = True):
    """get ML features in lo basis

    Args:
        cell : pyscf.pbc cell object
        custom_chkfile (string): mlgf chkfile object from generate.py
        C_ao_lo (np.float64, norb x norb): rotation matrix from AO to LO

    Returns:
        dict: modified mlf dictionary with lo basis features
    """
    if ao_labels is None:
        ao_labels = cell.ao_labels()

    mlf[f'C_ao_{basis_name}'] = C_ao_lo

    mo_energy = mlf['mo_energy']
    mo_coeff = mlf['mo_coeff']
    S_ao = mlf['ovlp']
    nkpts = S_ao.shape[0]
    nocc = mlf['nocc']
    nmo = len(mo_energy) 
    dm = mlf['dm_hf']

    mlf['hcore_nuc'] = cell.pbc_intor('int1e_nuc', kpts = mlf['kpts']) 
    mlf['hcore_kin'] = cell.pbc_intor('int1e_kin', kpts = mlf['kpts']) 

    nelectron = nocc*2

    S_lo = rotate_rank3(S_ao, C_ao_lo)
    
            
    # feature 1: fock matrix
    fock = mlf['fock']
    fock_lo = rotate_rank3(fock, C_ao_lo)
    
    # WARNING: THIS DOES NOT WORK FOR NONORTH LO BASIS
    # assumes S_lo_inv = I
    dm_lo = rotate_rank3(dm, C_ao_lo, ovlp = S_ao)
   
    # feature 3 : hcore matrix
    hcore = mlf['hcore']
    hcore_lo = rotate_rank3(hcore, C_ao_lo)
    
    mlf[f'dm_{basis_name}'] = dm_lo
    mlf[f'fock_{basis_name}'] = fock_lo
    mlf[f'hcore_{basis_name}'] = hcore_lo

    SCS = np.asarray([np.linalg.multi_dot((C_ao_lo[i].T, S_ao[i])) for i in range(nkpts)])
    mlf[f'C_{basis_name}_ao'] = SCS # for fock matrix
    mlf[f'S_{basis_name}'] = S_lo

    mlf[f'hcore_nuc_{basis_name}'] = rotate_rank3(mlf['hcore_nuc'], C_ao_lo)
    mlf[f'hcore_kin_{basis_name}'] = rotate_rank3(mlf['hcore_kin'], C_ao_lo)

    if 'vj' in mlf.keys():
        vj_lo = rotate_rank3(mlf['vj'], C_ao_lo)
        mlf[f'vj_{basis_name}'] = vj_lo

    if 'vxc' in mlf.keys():
        vxc_lo = rotate_rank3(mlf['vxc'], C_ao_lo)
        mlf[f'vxc_{basis_name}'] = vxc_lo

    if 'vk' in mlf.keys():
        vk_lo = rotate_rank3(mlf['vk'], C_ao_lo)
        mlf[f'vk_{basis_name}'] = vk_lo

    true_keys = ['fock_true', 'vk_true', 'vj_true', 'vxc_true']
    for key in true_keys:
        if key in mlf.keys():
            mlf[f'{key}_{basis_name}'] = rotate_rank3(mlf[key], C_ao_lo)

    if 'dm_hf_true' in mlf.keys():
        mlf[f'dm_true_{basis_name}'] = rotate_rank3(mlf['dm_hf_true'], C_ao_lo, ovlp = S_ao)

    if f'vj_{basis_name}' in mlf.keys():
        mlf[f'hcore+vj_{basis_name}'] = mlf[f'hcore_{basis_name}'] + mlf[f'vj_{basis_name}']
        mlf[f'vj_{basis_name}/nelec'] = mlf[f'vj_{basis_name}'] / cell.nelectron    


    if f'vj_true_{basis_name}' in mlf.keys():
        mlf[f'hcore+vj_true_{basis_name}'] = mlf[f'hcore_{basis_name}'] + mlf[f'vj_true_{basis_name}']
        mlf[f'vj_true_{basis_name}/nelec'] = mlf[f'vj_true_{basis_name}'] / cell.nelectron   

    mlf[f'hcore_{basis_name}/nelec'] = mlf[f'hcore_{basis_name}'] / cell.nelectron

    # extra simple orbital features

    df_mol = get_orbtypes_df(cell, ao_labels)
    mlf['inds_core'] = get_core_orbital_indices(ao_labels)
    mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = np.diag(df_mol['principal']), np.diag(df_mol['angular'])
    mlf[f'atomic_charge_{basis_name}'] = np.asarray([get_saiao_charges(df_mol, mlf[f'dm_{basis_name}'][i]) for i in range(nkpts)])
    mlf[f'boys_{basis_name}'] = np.asarray([get_saiao_locality(cell, mlf[f'C_ao_{basis_name}'][i]) for i in range(nkpts)])

    charges = np.array(df_mol['nuclear_charge'])
    mlf['nuclear_charge'] = np.abs(charges[:, None] - charges[None, :])
    np.fill_diagonal(mlf['nuclear_charge'], charges)
    mlf['nuclear_charge'] = np.asarray([mlf['nuclear_charge'] for i in range(nkpts)])
    coordinates = np.array(df_mol[['coordinate_x', 'coordinate_y', 'coordinate_z']])
    mlf['atomic_distance'] = distance_matrix(coordinates, coordinates)
    mlf['atomic_distance'] = np.asarray([mlf['atomic_distance'] for i in range(nkpts)])
    mlf['n_atoms'] = len(cell._atom)
    mlf['atomic_index'] = list(df_mol['atm_id'])

    return mlf

def get_vk_ml(mlf_chkfile, pickle_file):
    """get ML sigma

    Args:
        mlf_chkfile (str): DFT calculation file
        pickle_file (str): joblib file with model object

    Returns:
        np.complex64: self-energy
    """    
    if  '.joblib' in pickle_file: 
        model_obj = joblib.load(pickle_file)
    return model_obj.predict_vk(mlf_chkfile)

def get_vmo(mlf):
    """get MO basis vk

    Args:
        mlf: DFT calculation dictionary

    Returns:
        np.float64, np.float64: vk_mo, vxc_mo
    """    
    mo_coeff, vk, vmf = mlf['mo_coeff'], mlf['vk'], mlf['vxc']
    vk_mo = np.linalg.multi_dot((mo_coeff.T, vk, mo_coeff))
    vmf_mo = np.linalg.multi_dot((mo_coeff.T, vmf, mo_coeff))
    return vk_mo, vmf_mo

def get_triu_indices(n):
    # Cached to avoid repeated calls to np.triu_indices
    return np.triu_indices(n, k=1, m=None)

def unravel_rank3(mat):
    iu = get_triu_indices(mat.shape[0])
    if mat.ndim == 3:
        return mat[iu[0], iu[1], :]
    else:
        return mat[iu]

def find_subset_indices_numpy(fullset, subset, **kwargs):
    """get indices corresponding to subset in fullset, mainly for extracting AC indices 

    Args:
        fullset (np.float64)
        subset (np.float64)

    Raises:
        ValueError: if subset entry not in full set
        ValueError: finds multiple matches of a subset entry in full set

    Returns:
        np.array: indices
    """    
    assert(fullset.ndim == 1 and subset.ndim == 1)
    inds=[]
    for value in subset:
        match = np.flatnonzero(np.isclose(fullset, value, **kwargs))
        if match.size == 0:
            raise ValueError(f'Could not find {value} in {fullset}')
        elif match.size > 1:
            raise ValueError(f'Found multiple matches for {value} in {fullset}')
        inds.append(match[0])
    return np.array(inds, dtype=int)

# return GW in SAIAO basis from MO basis
# C^-1 = C^T (unitary in orthoganol basis ie SAIAO)
def gGW_mo_saiao_slow(gGW, C_mo_lo):
    gGW_saiao = np.zeros_like(gGW)
    for i in range(np.shape(gGW)[2]):
        gGW_saiao[:,:,i] = np.dot(C_mo_lo.T, gGW[:,:,i]).dot(C_mo_lo)
    return gGW_saiao

def gGW_mo_saiao(gGW, C_mo_lo):
    """GF rotation MO to local basis

    Args:
        gGW (np.array.complex64): greens function in MO basis
        C_mo_lo (np.array.float64): mo to local orbital basis rotation

    Returns:
        np.complex64: rotated GF
    """    
    
    return np.einsum('ji,jkn,kl->iln', C_mo_lo, gGW, C_mo_lo, optimize = True)

# return sigma in MO basis from localized orbital basis (SAAO or SAIAO)
def sigma_lo_mo(sigma_saiao, C_lo_mo):
    """GF rotation local basis to MO basis

    Args:
        gGW (np.array.complex64): greens function in local basis
        C_mo_lo (np.array.float64): local basis to MO basis rotation

    Returns:
        np.complex64: rotated GF
    """    
    return np.einsum('ji,jkn,kl->iln', C_lo_mo, sigma_saiao, C_lo_mo, optimize = True)

def get_g0(omega, mo_energy, eta):
    """get G0 (MF level) in MO basis

    Args:
        omega (np.array float64): frequencies on which to compute G0
        mo_energy (np.array float64): KS orbital energies
        eta (float): band-broadening

    Returns:
        np.array complex64: G0
    """    

    nmo = len(mo_energy)
    nw = len(omega)
    gf0 = np.zeros((nmo,nmo,nw),dtype=np.complex128)
    for iw in range(nw):
        gf0[:,:,iw] = np.diag(1.0/(omega[iw]+1j*eta - mo_energy))

    return gf0


def get_custom_freq_gfhf_features(mo_energy, fock_saiao, C_saiao_mo, omega, mlf_chkfile = ''):
    """diagonal dynamical features G0 and hyb

    Args:
        mo_energy (np.array float64): KS orbital energies
        fock_saiao (np.array float64): fock matrix in SAIAO
        C_saiao_mo (np.array float64): saiao to mo rotation matrix
        omega (np.array complex64): omega on which to generate dyn features
        mlf_chkfile (str, optional): file str to print if error. Defaults to ''.

    Returns:
        _type_: _description_
    """    
    # GF in MO basis on (ef + iw_n)
    g0_mo = get_g0(omega, mo_energy, eta=0)
    nmo = len(mo_energy)
    
    # GF in SAIAO basis   
    g0_saiao = np.einsum('ij,jkn,lk->iln', C_saiao_mo, g0_mo, C_saiao_mo, optimize = True)

    assert_traces = np.linalg.norm(np.trace(g0_mo[:,:,0])-np.trace(g0_saiao[:,:,0]))
    if assert_traces > 1e-8:
        warnings.warn(f'Warning! Trace of g0_mo and g0_saiao should be same (trace invariant under cyclic permutations), trace diff on w0: {assert_traces}, mlf_chkfile: {mlf_chkfile}')

    fock_diagonal = np.diagonal(fock_saiao)
    g0_diagonal = np.diagonal(g0_saiao)
    hyb_saiao = np.zeros_like(g0_saiao)
    
    hyb_saiao[np.arange(nmo), np.arange(nmo), :] = (omega[:, None] - fock_diagonal - 1. / g0_diagonal).T
    return g0_saiao, hyb_saiao

# fast pinv
# def stack_pinverse(mat_rank3):
#     """Compute the inverse of matrices in an array of shape (N,N,M)"""
#     return np.linalg.pinv(mat_rank3.transpose(2,0,1)).transpose(1,2,0)

def fast_pinv_2d(mats, det_tol = 1e-20):
    """quickly computes pinv of many 2x2 matrices using determinant, uses np.linalg.pinv (i.e. SVD Moorse-Penrose inverse) when det is near 0.
    speedup over all np.linalg.pinv should be ~8-fold

    Args:
        mats (np.float64): shape N x 2 x 2, the stack of 2 x 2 matrices
        det_tol (float, optional): near-zero tolerance for defaulting to np.linalg.pinv. Defaults to 1e-20.

    Returns:
        Nx2x2 stack of the pseudo inverses
    """    
    # assumes the matrices are symmetric
    try:
        import numexpr as ne
        a, b, c = mats[:,0,0].copy(), mats[:,1,1].copy(), mats[:,0,1].copy()
        dets = ne.evaluate('a*b-c**2')
        mask = np.abs(dets) > det_tol
        mats[mask,0,0] = b
        mats[mask,1,1] = a
        mats[mask,0,1] = - c
        mats[mask,1,0] = - c
        mat_masked = mats[mask,:,:]
        det_masked = dets[mask][:, np.newaxis, np.newaxis]
        mats[mask,:,:] = ne.evaluate('mat_masked / det_masked')
        mats[~mask] = np.linalg.pinv(mats[~mask])
    except ImportError:
        # numpy when numexpr not available
        dets = mats[:,0,0]*mats[:,1,1] - mats[:,0,1]*mats[:,1,0]
        mask = np.abs(dets) > det_tol
        a = mats[mask,0,0].copy()
        mats[mask,0,0] = mats[mask,1,1].copy()
        mats[mask,1,1] = a
        mats[mask,0,1] = - mats[mask,0,1]
        mats[mask,1,0] = - mats[mask,1,0]
        mats[mask,:,:] = mats[mask,:,:] / dets[mask][:, np.newaxis, np.newaxis]
        mats[~mask] = np.linalg.pinv(mats[~mask])
    return mats

def get_hyb_off(fock_saiao, g0_saiao, omega, mask = None):
    """Get hyb[ij] dynamical features

    Args:
        fock_saiao (np.array float64): fock in SAIAO
        g0_saiao (np.array complex64): g0 in saiao
        omega (np.array complex64): omega points on which to calculate the feature
        mask (np.array float64, optional): nmo x nmo mask to avoid computing full hyb_off (i.e. if edges are screened out), defaults to ones, i.e. fully compute hyb_off
    """    
    

    norbs = g0_saiao.shape[0]
    iu, il = np.triu_indices(norbs, 1), np.tril_indices(norbs, -1)
    n_ij = len(iu[0])
    if mask is None:
        mask = np.ones(n_ij, dtype=bool)
    else:
        mask = mask[iu[0], iu[1]]
    nomega = len(omega)
    offdiag = g0_saiao[iu]
    diag1 = g0_saiao[iu[0], iu[0], :]
    diag2 = g0_saiao[iu[1], iu[1], :]
    hyb_off_saiao = np.zeros((n_ij, nomega), dtype=np.complex_)
    stacks = np.zeros((n_ij, nomega, 2, 2), dtype=np.complex_)

    omega_minus_fock =  - np.tile(fock_saiao[iu], (len(omega),1)).T 
    
    stacks[:, :, 0, 0] = diag1
    stacks[:, :, 1, 1] = diag2
    stacks[:, :, 0, 1] =  offdiag
    stacks[:, :, 1, 0] =  offdiag

    for w in range(nomega):
        g0_inv = fast_pinv_2d(stacks[mask,w,:,:])[:,0,1]
        hyb_off_saiao[mask, w] = omega_minus_fock[mask,w] - g0_inv
    
    result = np.zeros(g0_saiao.shape, dtype = np.complex128)
    result[iu] = hyb_off_saiao
    result[il] = result.transpose((1, 0, 2))[il]

    return result

def get_sigma_fit(full_sigma, full_freqs, selected_freqs, freq_tol = 1.0e-8):
    """subset by frequencies

    Args:
        full_sigma (_type_): _description_
        full_freqs (_type_): _description_
        selected_freqs (_type_): _description_
        freq_tol (_type_, optional): _description_. Defaults to 1.0e-8.

    Returns:
        complex64: iomega subset of sigma
    """    
    assert(len(full_freqs)==full_sigma.shape[-1])
    sigma_subset = full_sigma[:,:,find_subset_indices_numpy(full_freqs, np.imag(selected_freqs), atol=freq_tol)]
    return sigma_subset.copy()

def get_chk_saiao(mf, fock_ao, minao = 'minao', force_eigv_direction = False, val_core_dats = None):
    """wrapper for localized IAO transform and symmetrization rotations 

    Args:
        mf (pyscf.mf): mean field object like DFT
        fock_ao (float64): ao basis fock
        minao (str, optional): which definition of MINAO to use. Defaults to 'minao'.
        force_eigv_direction (bool, optional): force the eigenvalue direction. Defaults to False.
        val_core_dats (list, optional): data for redefining MINAO native to pyscf. Defaults to None.

    Returns:
        tuple: C_ao_iao, C_iao_saiao, fock_iao
    """    
    if val_core_dats is None:
        C_ao_iao = get_C_ao_iao(mf, minao=minao)
    else:
        C_ao_iao = get_C_ao_iao(mf, minao=minao, minao_val=val_core_dats[0], minao_core=val_core_dats[1])
    fock_iao = reduce(np.dot, (C_ao_iao.T, fock_ao, C_ao_iao))
    C_iao_saiao = get_saao(mf.mol, fock_iao, force_eigv_direction = force_eigv_direction)
    
    assert(np.linalg.norm(np.dot(C_iao_saiao.T, C_iao_saiao)-np.eye(fock_iao.shape[0]))<1e-8)
    return C_ao_iao, C_iao_saiao, fock_iao

def get_chk_saiao_vsap(mf, fock_ao, minao = 'minao', force_eigv_direction = False, val_core_dats = None,
                       sap_type = 'dft', reorder_saiao = True):
    """wrapper for localized IAO from Vsap transform and symmetrization rotations 

    Args:
        mf (pyscf.mf): mean field object like DFT
        fock_ao (float64): ao basis fock
        minao (str, optional): which definition of MINAO to use. Defaults to 'minao'.
        force_eigv_direction (bool, optional): force the eigenvalue direction. Defaults to False.
        val_core_dats (list, optional): data for redefining MINAO native to pyscf. Defaults to None.

    Returns:
        tuple: C_ao_iao, C_iao_saiao, fock_iao
    """    
    if val_core_dats is None:
        C_ao_iao = get_C_ao_iao_vsap(mf, minao=minao, sap_type=sap_type)
    else:
        C_ao_iao = get_C_ao_iao_vsap(mf, minao=minao, minao_val=val_core_dats[0], minao_core=val_core_dats[1], sap_type=sap_type)
    fock_iao = reduce(np.dot, (C_ao_iao.T, fock_ao, C_ao_iao))
    C_iao_saiao = get_saao(mf.mol, fock_iao, force_eigv_direction = force_eigv_direction)
    if reorder_saiao:
        if val_core_dats is None:
            C_iao_saiao_reorder = reorder_iaofirst(mf, C_iao_saiao, minao=minao)
        else:
            C_iao_saiao_reorder = reorder_iaofirst(mf, C_iao_saiao, minao=minao, minao_val=val_core_dats[0], minao_core=val_core_dats[1])
        C_iao_saiao = C_iao_saiao_reorder
    
    assert(np.linalg.norm(np.dot(C_iao_saiao.T, C_iao_saiao)-np.eye(fock_iao.shape[0]))<1e-8)
    return C_ao_iao, C_iao_saiao, fock_iao

def get_chk_saao(mol, fock):  
    """only has symmetrization, no IAO

    Args:
        mol (pyscf.mol):
        fock (float64): AO fock matrix

    Returns:
        float64: AO to SAAO rotation
    """      
    C_ao_saao = get_saao(mol, fock)
    return C_ao_saao

def get_core_orbital_indices(ao_labels):
    """get indicies of core orbs

    Args:
        mol (pyscf.mol):

    Returns:
        list: core index
    """    
    from mlvk.misc.config import config as core_config
    inds_core = []
    for i, label in enumerate(ao_labels):
        for core in core_config:
            # support format like 'S (1s|2s|2p)'
            # to S 1s, S 2s, S 2p
            if '(' in core and ')' in core:
                core_split = core.split('(')
                core_prefix = core_split[0].strip()
                core_orbs = core_split[1].split(')')[0].split('|')
                for orb in core_orbs:
                    if core_prefix in label and orb in label:
                        inds_core.append(i)
            else:
                if core in label:
                    inds_core.append(i)
    return inds_core

def exclude_core_ftrs(feat, inds_core, rank2 = True):
    """removes core orb features

    Args:
        feat (float64): feature array
        inds_core (list): _description_
        rank2 (bool, optional): is the feature only rank2 (or rank3). Defaults to True.

    Returns:
        float64: feature array with core orbitals removed
    """    
    feat_new = np.delete(feat, inds_core, axis=0)
    if rank2: feat_new = np.delete(feat_new, inds_core, axis=1)
    return feat_new

def get_rsq(y_pred, y_true):
    """get rsq value between true and pred

    Args:
        y_pred (float64): pred
        y_true (float64): true

    Returns:
        float: rsq value
    """    
    ybar = np.mean(y_true)
    SST = np.sum((y_true - ybar)**2)
    SSReg = np.sum((y_pred - y_true)**2)
    return 1-SSReg/SST

def get_sigma_from_ml(sigma_ii, sigma_ij):
    """reshaping function for KRR

    Args:
        sigma_ii (float64): diag elements
        sigma_ij (float64): offdiag elements

    Returns:
        complex64: norb x norb x nomega
    """    
    nomega = sigma_ii.shape[-1]//2 
    norbs = sigma_ii.shape[0]
    sigma_ml_saiao = np.zeros((norbs, norbs, nomega), dtype=np.complex_)
    iu, il = np.triu_indices(norbs, 1), np.tril_indices(norbs, -1)
    for iw in range(nomega):     
        sigma_ml_saiao_iw = np.diag(sigma_ii[:,iw] + 1j*sigma_ii[:,nomega+iw])
        sigma_ml_saiao_iw[iu] = sigma_ij[:,iw] + 1j*sigma_ij[:,nomega+iw]
        sigma_ml_saiao_iw[il] = sigma_ml_saiao_iw.T[il]
        sigma_ml_saiao[:, :, iw] = sigma_ml_saiao_iw

    return sigma_ml_saiao

def rotate_sigma_uncertainty(sigma_uncertainty, C_lo_mo):
    """rotate uncertainty tensor from SAIAO to MO

    Args:
        sigma_uncertainty (complex64): uncertainty tensor
        C_lo_mo (float64): SAIAO to MO rotation

    Returns:
        complex64: norb x norb x nomega
    """    
    C_lo_mo = C_lo_mo**2
    return np.sqrt(sigma_lo_mo(sigma_uncertainty.real**2, C_lo_mo))+1j*np.sqrt(sigma_lo_mo(sigma_uncertainty.imag**2, C_lo_mo))

# def set_2orb_inds(mat, inds1, ind2, val):
#     idx = np.array(list(itertools.product(inds1, ind2)))
#     mat[tuple(np.array(idx).T)] = val # core-core
#     idx = np.array(list(itertools.product(ind2, inds1)))
#     mat[tuple(np.array(idx).T)] = val # core-core
#     return mat
    
def get_orb_type(mol, dm, virtual_tol = 1e-10):
    types = np.empty(dm.shape, dtype = 'int')
    
    inds_core = get_core_orbital_indices(mol.ao_labels())
    
    pao_cond = np.abs(np.diagonal(dm)) < virtual_tol
    inds_pao = np.where(pao_cond)[0]
    inds_iao = np.where(~pao_cond)[0]
    inds_iao = list(set(list(inds_iao)) - set(list(inds_core)))
     
    # types = set_2orb_inds(types, inds_core, inds_core, 0) # core-core
    # types = set_2orb_inds(types, inds_core, inds_iao, 1) # core-iao
    # types = set_2orb_inds(types, inds_core, inds_pao, 2) # core-pao
    # types = set_2orb_inds(types, inds_iao, inds_iao, 3) # iao-iao
    # types = set_2orb_inds(types, inds_iao, inds_pao, 4) # iao-pao
    # types = set_2orb_inds(types, inds_pao, inds_pao, 5) # pao-pao
    
    types[inds_pao, inds_pao] = 2 # pao
    types[inds_iao, inds_iao] = 1 # iao
    types[inds_core, inds_core] = 0 # core
    return types

def get_orb_principal(mol, dm):
    """converts princ number in pyscf mol string to an integer

    FIXME
    """    
    types = np.empty(dm.shape, dtype = 'int')
    inds = []
    chars = [' 1', ' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8']
    for char in chars:
        inds.append(list(mol.search_ao_label(char)))
    
    return np.diag(inds)

def get_ang_from_letter(letter):
    """converts angular letter in pyscf mol string to an integer

    Args:
        letter (str): spdf, etc

    Returns:
        int: l, -1 if failure
    """    
    if letter == 's':
        return 0
    if letter == 'p':
        return 1
    if letter == 'd':
        return 2
    if letter == 'f':
        return 3
    return -1

# def get_orbtypes(mol):
#     """orbital quantum numbers n and l

#     Args:
#         mol (pyscf.mol)

#     Returns:
#         tuple: n, l (both norb x norb)
#     """    
#     types_princ, types_ang = np.zeros(mol.nao), np.zeros(mol.nao)
#     types_princ[:], types_ang[:] = -1, -1
#     labs = mol.ao_labels()
#     labs = [l.split(' ') for l in labs]
#     for i in range(mol.nao):
#         lab = labs[i]
#         types_princ[i] = int(lab[2][0])-1
#         types_ang[i] = get_ang_from_letter(lab[2][1])
#     return np.diag(types_princ).astype('int'), np.diag(types_ang).astype('int')

def get_orbtypes_df(mol, ao_labels = None):
    """get orbital metadata as pandas table

    Args:
        mol (pyscf.mol)
        ao_labels (list of str): AO labels

    Returns:
        pd.DataFrame: metadata
    """    
    if ao_labels is None:
        ao_labels = mol.ao_labels()
    df = pd.Series(ao_labels).str.split(pat = ' ', n = 2, expand = True)
    df.columns = ['atm_id', 'atm_symbol', 'orbital_name']
    df['atm_id'] = df['atm_id'].astype('int')
    df['principal'] = df['orbital_name'].str[0].astype('int')
    df['angular'] = -1
    df.loc[df['orbital_name'].str.contains('s'), 'angular'] = 0
    df.loc[df['orbital_name'].str.contains('p'), 'angular'] = 1
    df.loc[df['orbital_name'].str.contains('d'), 'angular'] = 2
    df.loc[df['orbital_name'].str.contains('f'), 'angular'] = 3
    ids = np.unique(df['atm_id'])
    atom_metadata = pd.DataFrame({'atm_id' : ids})
    atom_metadata['nuclear_charge'] = [mol.atom_charge(i) for i in list(ids)]
    atom_metadata['coordinate_x'] = [a[1][0] for a in mol._atom]
    atom_metadata['coordinate_y'] = [a[1][1] for a in mol._atom]
    atom_metadata['coordinate_z'] = [a[1][2] for a in mol._atom]
    df = df.merge(atom_metadata, on = 'atm_id')
    return df 

def get_saiao_charges(df_mol, dm_saiao):
    """IAO (SAIAO) based charges 

    Args:
        df_mol (pd.dataframe f): rom get_orbtypes_df(), has nuclear_charge column
        dm_saiao (np.array float64): SAIAO density matrix

    Returns:
        SAIAO charges: norb x norb
    """    
    df_mol['elec_density'] = np.diagonal(dm_saiao)
    df_mol['atm_charge_density'] = df_mol['nuclear_charge'].astype('float64') - df_mol.groupby(by = 'atm_id')['elec_density'].transform('sum')
    return np.diag(df_mol['atm_charge_density'])

def get_saiao_locality(mol, C_ao_saiao):
    """Boys localization feature for SAIAO: f_i = < i | r^2 | i> - <i | r | i> ^ 2

    Args:
        mol (pyscf.mol)
        C_ao_saiao (np.array float64): AO to SAIAO rotation

    Returns:
        boys: norb x norb matrix
    """    
    # 
    r1 = mol.intor_symmetric('int1e_r')
    r2 = mol.intor_symmetric('int1e_r2')
    r1_saiao = np.einsum('ui, xuv, vi -> xi', C_ao_saiao, r1, C_ao_saiao, optimize = True)
    r2_saiao = np.einsum('ui, uv, vi -> i', C_ao_saiao, r2, C_ao_saiao, optimize = True)
    boys = r2_saiao - np.einsum('xi, xi -> i', r1_saiao, r1_saiao, optimize = True)
    return np.diag(boys)


def get_downfold_features(mol, mlf, minao='minao', basis_name = 'saiaovsap', ao_labels = None):
    """get ML features from basis_name to downfold basis
    """
    from mlvk.workflow import ham_downfolding
    assert f'focksap_{basis_name}' in mlf.keys()
    S_lo = mlf[f'S_{basis_name}']
    assert np.allclose(mlf[f'S_{basis_name}'], np.eye(S_lo.shape[0])), "S_lo is not identity, cannot use downfold features!"   

    from pyscf.lo.iao import reference_mol
    pmol = reference_mol(mol, minao=minao)
    if ao_labels is None:
        ao_labels = pmol.ao_labels()
    nB2 = len(pmol.ao_labels())
    x_exact = ham_downfolding.get_X_matrix(mlf[f'fock_true_{basis_name}'], nB2, list(range(nB2)))[0]
    fock_true_downfold_exact = ham_downfolding.get_downfold_matrix(mlf[f'fock_true_{basis_name}'], x_exact)
    mlf[f'fock_true_{basis_name}_downfold_exact'] = fock_true_downfold_exact
    mlf[f'S_{basis_name}_downfold_exact'] = ham_downfolding.get_downfold_matrix(mlf[f'S_{basis_name}'], x_exact)
    mlf[f'dipole_x_{basis_name}_downfold_exact'] = ham_downfolding.get_downfold_matrix(mlf[f'dipole_x_{basis_name}'], x_exact)
    mlf[f'dipole_y_{basis_name}_downfold_exact'] = ham_downfolding.get_downfold_matrix(mlf[f'dipole_y_{basis_name}'], x_exact)
    mlf[f'dipole_z_{basis_name}_downfold_exact'] = ham_downfolding.get_downfold_matrix(mlf[f'dipole_z_{basis_name}'], x_exact)
    e, c = scipy.linalg.eigh(fock_true_downfold_exact, mlf[f'S_{basis_name}_downfold_exact'])
    dm_downfold_exact = 2.0 * c[:,:mol.nelectron//2] @ c[:,:mol.nelectron//2].T
    mlf[f'dm_hf_true_{basis_name}_downfold_exact'] = dm_downfold_exact
    mlf['X_downfold_exact'] = x_exact
    x_exact = None

    x_sap = ham_downfolding.get_X_matrix(mlf[f'focksap_{basis_name}'], nB2, list(range(nB2)))[0]
    mlf['X_downfold_vsap'] = x_sap
    for key in ['hcore_nuc', 'hcore_kin', 'vsap', 'hcore', 'focksap', 'S', 'fock', 'boys', 'atomic_charge']:
        key_lo = f'{key}_{basis_name}'
        mlf[f'{key_lo}_downfold'] = ham_downfolding.get_downfold_matrix(mlf[key_lo], x_sap)
    f_AA, f_hyb, f_AA_R, f_hyb_R = ham_downfolding.get_downfold_matrix_extra(mlf[f'fock_{basis_name}'], x_sap)
    mlf[f'fock_AA_{basis_name}_downfold'] = f_AA
    mlf[f'fock_hyb_{basis_name}_downfold'] = f_hyb
    mlf[f'fock_AA_R_{basis_name}_downfold'] = f_AA_R
    mlf[f'fock_hyb_R_{basis_name}_downfold'] = f_hyb_R
    
    e, c = scipy.linalg.eigh(mlf[f'focksap_{basis_name}_downfold'], mlf[f'S_{basis_name}_downfold'])
    mo_occ = np.zeros_like(e)
    mo_occ[:mol.nelectron//2] = 2
    dm_downfold = c @ np.diag(mo_occ) @ c.T
    mlf[f'dm_{basis_name}_downfold'] = dm_downfold

    for key in ['vj', 'vxc', 'vk']:
        key_lo = f'{key}_{basis_name}'
        if key in mlf.keys():
            mlf[f'{key_lo}_downfold'] = ham_downfolding.get_downfold_matrix(mlf[key_lo], x_sap)

    # true_keys = ['fock_true', 'vk_true', 'vj_true', 'vxc_true']
    # for key in true_keys:
    #     if key in mlf.keys():
    #         mlf[f'{key}_{basis_name}_downfold'] = ham_downfolding.get_downfold_matrix(mlf[f'{key}_{basis_name}'], x_sap)


    if f'vj_{basis_name}_downfold' in mlf.keys():
        mlf[f'hcore+vj_{basis_name}_downfold'] = mlf[f'hcore_{basis_name}_downfold'] + mlf[f'vj_{basis_name}_downfold']
        mlf[f'vj_{basis_name}_downfold/nelec'] = mlf[f'vj_{basis_name}_downfold'] / mol.nelectron    

    mlf[f'hcore_{basis_name}_downfold/nelec'] = mlf[f'hcore_{basis_name}_downfold'] / mol.nelectron


    if f'fock_true_{basis_name}_downfold_exact' in mlf.keys() and f'fock_{basis_name}_downfold' in mlf.keys():
        mlf[f'delta_fock_{basis_name}_downfold'] = mlf[f'fock_true_{basis_name}_downfold_exact'] - mlf[f'fock_{basis_name}_downfold']

    # extra simple orbital features
    df_mol = get_orbtypes_df(pmol, ao_labels)
    # mlf[f'atomic_charge_{basis_name}_downfold'] = get_saiao_charges(df_mol, mlf[f'dm_{basis_name}_downfold'])
    # mlf[f'boys_{basis_name}_downfold'] = get_saiao_locality(mol, mlf[f'C_ao_{basis_name}'])[:nB2, :nB2]
    # overwrite some features with downfolded versions
    mlf['inds_core'] = get_core_orbital_indices(ao_labels)
    mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = np.diag(df_mol['principal']), np.diag(df_mol['angular'])

    return mlf

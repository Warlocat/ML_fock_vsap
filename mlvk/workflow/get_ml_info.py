import joblib
import numpy as np
import scipy
from pyscf import lib, scf
from mlvk.lib.ml_helper import get_saiao_charges, get_orbtypes_df

def predict_veff(inp_file, model_joblib_file, return_model = False):
    model_obj = joblib.load(model_joblib_file)
    veff_ml_lo = model_obj.predict_vk(inp_file)
    if return_model:
        return veff_ml_lo, model_obj

def make_rdm1_docc(mo_coeff, nocc):
    return 2*mo_coeff[:,:nocc] @ (mo_coeff[:,:nocc].T)

def get_lowdin_charges(mol, dm):
    df_mol = get_orbtypes_df(mol)
    df_mol['lowdin_charge'] = np.diagonal(get_saiao_charges(df_mol, dm))
    df_mol = df_mol[['atm_id', 'lowdin_charge', 'nuclear_charge']].drop_duplicates()
    return np.array(df_mol['lowdin_charge']-df_mol['nuclear_charge'])

def get_dipole(mol, dm_ao):
    """dipole

    Parameters
    ----------
    mol : pyscf mol object
    dm_ao : float64 2d array
        AO rdm1

    Returns
    -------
    float
        dipole
    """    
    dip = scf.hf.dip_moment(mol, dm_ao, verbose = 0)
    return np.linalg.norm(dip)

def get_elec_dipole_vec(mol, dm_ao):
    dipole_ints = -mol.intor_symmetric('int1e_r', comp=3)
    dip = np.einsum('xij,ji->x', dipole_ints, dm_ao).real
    return dip

# def get_dipole_lo(dm_lo, C_ao_lo, mol, origin = None):
#     from pyscf.data import nist

#     if origin is None:
#         origin = np.zeros(3)

#     charges = mol.atom_charges()
#     coords  = mol.atom_coords()

#     with mol.with_common_orig(origin):
#         ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    
#     lo_dipole = np.einsum('ji,xjk,kl->xil', C_ao_lo, ao_dip, C_ao_lo, optimize = True)
#     el_dip = np.einsum('xij,ji->x', lo_dipole, dm_lo).real
#     nucl_dip = np.einsum('i,ix->x', charges, coords - origin[None, :])
#     mol_dip = nucl_dip - el_dip
#     mol_dip *= nist.AU2DEBYE
#     return np.linalg.norm(mol_dip)

def get_quadrupole(mol, dm_ao):
    """dipole

    Parameters
    ----------
    mol : pyscf mol object
    dm_ao : float64 2d array
        AO rdm1

    Returns
    -------
    float
        dipole
    """    
    traceless_quad = scf.hf.quad_moment(mol, dm_ao, verbose = 0)
    evs, _ = np.linalg.eigh(traceless_quad)
    return (evs[-1] - evs[0]) / 2

def get_qudrupole_elec_vec(mol, dm_ao):
    quad_ints = mol.intor_symmetric("int1e_rr", comp=9)
    quad = np.einsum('xij,ij->x', quad_ints, dm_ao).real
    return quad

def validate_model_mol(chk_file, model_joblib_file):
    print(chk_file, flush = True)
    validation_results = {}

    veff_ml_lo, model_obj = predict_veff(chk_file, model_joblib_file, return_model = True)
    idx_fock = getattr(model_obj, 'idx_fock', 0)
    fock_ml_lo = veff_ml_lo[:,:,idx_fock]
    
    mlf_ftr = model_obj.pdata

    basis = model_obj.basis.split('_')[0]
    C_ao_lo = mlf_ftr[f'C_ao_{basis}'].copy()
    C_lo_ao = mlf_ftr[f'C_{basis}_ao'].copy()

    if 'delta' in model_obj.targets[0]:
        fock_ml_lo += model_obj.pdata[f'fock_{basis}']

    fock_ml_ao = np.linalg.multi_dot((C_lo_ao.T, fock_ml_lo, C_lo_ao)) # in learning basis
    

    mlf_true = lib.chkfile.load(chk_file, 'mlf')
    mlf_true['hcore+vj'] = mlf_true['vj'] + mlf_true['hcore']
    S_ao = mlf_true['ovlp']
    S_lo = mlf_ftr[f'S_{basis}']

    fock_true_ao = mlf_true['fock'].copy()
    fock_true_lo = np.linalg.multi_dot((C_ao_lo.T, fock_true_ao, C_ao_lo)) # in learning basis
    
    fock_mae_lo = np.mean(np.abs(fock_true_lo - fock_ml_lo))
    fock_mae_ao = np.mean(np.abs(fock_true_ao- fock_ml_ao))

    validation_results['fock_lo_mae'] = fock_mae_lo
    validation_results['fock_ao_mae'] = fock_mae_ao

    mo_ml, mo_coeff_ml = scipy.linalg.eigh(fock_ml_ao, S_ao)
    validation_results['mo_energy_ml'] = mo_ml
    validation_results['mo_energy_mae'] = np.mean(np.abs(mo_ml - mlf_true['mo_energy']))
    
    nocc = mlf_true['nocc']
    validation_results['homo_ml'] = mo_ml[nocc-1]
    validation_results['homo_true'] = mlf_true['mo_energy'][nocc-1]

    validation_results['lumo_ml'] = mo_ml[nocc]
    validation_results['lumo_true'] = mlf_true['mo_energy'][nocc]

    # nonorth basis
    if basis not in ['salao', 'lao', 'saiao', 'iao', 'mo']:
        S_lo_inv = np.linalg.inv(S_lo)
        SCS = np.linalg.multi_dot((S_lo_inv, C_ao_lo.T, S_ao))
    # orthbasis
    else:
        SCS = np.linalg.multi_dot((C_ao_lo.T, S_ao))

    dm_ml_ao = make_rdm1_docc(mo_coeff_ml, mlf_true['nocc'])
    dm_true_ao = mlf_true['dm_hf']
    
    dm_ml_lo = np.linalg.multi_dot((SCS, dm_ml_ao, SCS.T))
    dm_true_lo = np.linalg.multi_dot((SCS, dm_true_ao, SCS.T))
    
    validation_results['ddm_lo'] = np.mean(np.abs(dm_ml_lo - dm_true_lo))
    validation_results['ddm_ao'] = np.mean(np.abs(dm_ml_ao - dm_true_ao))
    validation_results['lowdin_charge_ml'] = get_lowdin_charges(mlf_ftr.mol, dm_ml_lo)
    validation_results['lowdin_charge_true'] = get_lowdin_charges(mlf_ftr.mol, dm_true_lo)

    validation_results['lowdin_charge_mae'] = np.mean(np.abs(validation_results['lowdin_charge_ml'] - validation_results['lowdin_charge_true']))
    # relative error
    validation_results['lowdin_charge_rae'] = np.mean(np.abs((validation_results['lowdin_charge_ml'] - validation_results['lowdin_charge_true']) / validation_results['lowdin_charge_true']))


    validation_results['dipole_ml'] = get_dipole(mlf_ftr.mol, dm_ml_ao)
    # validation_results['dipole_lo_ml'] = get_dipole_lo(dm_ml_lo, C_ao_lo, mlf_ftr.mol)
    validation_results['dipole_true'] = get_dipole(mlf_ftr.mol, dm_true_ao)

    dip_elec_ml = get_elec_dipole_vec(mlf_ftr.mol, dm_ml_ao)
    dip_elec_true = get_elec_dipole_vec(mlf_ftr.mol, dm_true_ao)
    print("elec dipole ml:", dip_elec_ml)
    print("elec dipole true:", dip_elec_true)
    validation_results['dipole_elec_error'] = np.linalg.norm(dip_elec_ml - dip_elec_true)

    validation_results['quadrupole_ml'] = get_quadrupole(mlf_ftr.mol, dm_ml_ao)
    validation_results['quadrupole_true'] = get_quadrupole(mlf_ftr.mol, dm_true_ao)

    validation_results['quadrupole_elec_error'] = np.linalg.norm(get_qudrupole_elec_vec(mlf_ftr.mol, dm_ml_ao) - get_qudrupole_elec_vec(mlf_ftr.mol, dm_true_ao))

    validation_results['dm_ml'] = dm_ml_ao
    

    # for i, target in enumerate(model_obj.targets):
    #     this_target = target.split('_')[0]
    #     diff_lo = veff_ml_lo[:,:,i] - np.linalg.multi_dot((C_ao_lo.T, mlf_true[this_target], C_ao_lo)) # in learning basis
    #     v_ml_ao = np.linalg.multi_dot((C_lo_ao.T, veff_ml_lo[:,:,i], C_lo_ao))
    #     diff_ao = v_ml_ao - mlf_true[this_target] # in ao basis 

    #     validation_results[f'{this_target}_mae_lo'] = np.mean(np.abs(diff_lo))
    #     validation_results[f'{this_target}_mae_ao'] = np.mean(np.abs(diff_ao))

    #     validation_results[f'etot_{this_target}_ml'] = np.einsum('ij,ji', v_ml_ao, dm_ml_ao)
    #     validation_results[f'etot_{this_target}_true'] = np.einsum('ij,ji', mlf_true[this_target], dm_true_ao)

    return validation_results



def validate_model_mol_downfold(chk_file, model_joblib_file):
    print(chk_file, flush = True)
    validation_results = {}

    veff_ml_lo, model_obj = predict_veff(chk_file, model_joblib_file, return_model = True)
    idx_fock = getattr(model_obj, 'idx_fock', 0)
    fock_ml_lo = veff_ml_lo[:,:,idx_fock]
    
    mlf_ftr = model_obj.pdata

    basis = model_obj.basis.split('_')[0]
    C_ao_lo = mlf_ftr[f'C_ao_{basis}'].copy()
    C_lo_ao = mlf_ftr[f'C_{basis}_ao'].copy()

    if 'delta' in model_obj.targets[0]:
        fock_ml_lo += model_obj.pdata[f'fock_{basis}_downfold']

    fock_true_lo = mlf_ftr[f'fock_true_{basis}_downfold_exact']
    S_true_lo = mlf_ftr[f'S_{basis}_downfold_exact']
    fock_mae_lo = np.mean(np.abs(fock_true_lo - fock_ml_lo))
    validation_results['fock_lo_mae'] = fock_mae_lo

    mlf_true = lib.chkfile.load(chk_file, 'mlf')
    S_lo = mlf_ftr[f'S_{basis}_downfold']

    mo_ml_dwnfld, mo_coeff_ml_dwnfld = scipy.linalg.eigh(fock_ml_lo, S_true_lo)
    mo_true_dwnfld, mo_coeff_true_dwnfld = scipy.linalg.eigh(fock_true_lo, S_true_lo)
    mo_true_full, mo_coeff_true_full = scipy.linalg.eigh(mlf_true['fock'], mlf_true['ovlp'])

    validation_results['mo_energy_ml'] = mo_ml_dwnfld
    validation_results['mo_energy_dwnfld_mae'] = np.mean(np.abs(mo_ml_dwnfld - mo_true_dwnfld))
    validation_results['mo_energy_aae'] = np.mean(np.abs(mo_ml_dwnfld - mo_true_full[:len(mo_ml_dwnfld)]))
    
    nocc = mlf_true['nocc']
    validation_results['homo_ml'] = mo_ml_dwnfld[nocc-1]
    validation_results['homo_true'] = mo_true_full[nocc-1]
    validation_results['homo_true_dwnfld'] = mo_true_dwnfld[nocc-1]

    validation_results['lumo_ml'] = mo_ml_dwnfld[nocc]
    validation_results['lumo_true'] = mo_true_full[nocc]
    validation_results['lumo_true_dwnfld'] = mo_true_dwnfld[nocc]

    

    S_ao = mlf_true['ovlp']
    # assume orth basis !
    SCS = np.linalg.multi_dot((C_ao_lo.T, S_ao))

    # convert to original coeff for properties
    x_mat_vsap = mlf_ftr['X_downfold_vsap']
    # x_mat_vsap = mlf_ftr['X_downfold_exact']
    nB2 = x_mat_vsap.shape[1]
    print("Xnorm", np.linalg.norm(x_mat_vsap))
    s_prime = np.eye(nB2) + x_mat_vsap.T @ x_mat_vsap
    from pyscf.x2c import x2c
    r_mat = x2c._invsqrt(s_prime)
    mo_coeff_A = r_mat @ mo_coeff_ml_dwnfld
    mo_coeff_B = x_mat_vsap @ mo_coeff_A
    mo_coeff_ml = np.vstack((mo_coeff_A, mo_coeff_B))

    c_test = C_ao_lo @ mo_coeff_ml
    fock_test = c_test.T @ mlf_true['fock'] @ c_test
    print("X -> Fdiag",np.linalg.norm(fock_test - fock_test.diagonal() * np.eye(fock_test.shape[0])))
    ovlp_test = c_test.T @ mlf_true['ovlp'] @ c_test
    print("X -> S", np.linalg.norm(ovlp_test - np.eye(ovlp_test.shape[0])))

    dm_ml_ao = make_rdm1_docc(C_ao_lo @ mo_coeff_ml, mlf_true['nocc'])
    dm_true_ao = mlf_true['dm_hf']
    
    dm_ml_lo = np.linalg.multi_dot((SCS, dm_ml_ao, SCS.T))
    dm_true_lo = np.linalg.multi_dot((SCS, dm_true_ao, SCS.T))
    
    validation_results['ddm_lo'] = np.mean(np.abs(dm_ml_lo - dm_true_lo))
    validation_results['ddm_ao'] = np.mean(np.abs(dm_ml_ao - dm_true_ao))
    validation_results['lowdin_charge_ml'] = get_lowdin_charges(mlf_ftr.mol, dm_ml_lo)
    validation_results['lowdin_charge_true'] = get_lowdin_charges(mlf_ftr.mol, dm_true_lo)

    validation_results['lowdin_charge_mae'] = np.mean(np.abs(validation_results['lowdin_charge_ml'] - validation_results['lowdin_charge_true']))
    # relative error
    validation_results['lowdin_charge_rae'] = np.mean(np.abs((validation_results['lowdin_charge_ml'] - validation_results['lowdin_charge_true']) / validation_results['lowdin_charge_true']))


    validation_results['dipole_ml'] = get_dipole(mlf_ftr.mol, dm_ml_ao)
    # validation_results['dipole_lo_ml'] = get_dipole_lo(dm_ml_lo, C_ao_lo, mlf_ftr.mol)
    validation_results['dipole_true'] = get_dipole(mlf_ftr.mol, dm_true_ao)
    validation_results['dipole_elec_error'] = np.linalg.norm(get_elec_dipole_vec(mlf_ftr.mol, dm_ml_ao) - get_elec_dipole_vec(mlf_ftr.mol, dm_true_ao))
    validation_results['quadrupole_ml'] = get_quadrupole(mlf_ftr.mol, dm_ml_ao)
    validation_results['quadrupole_true'] = get_quadrupole(mlf_ftr.mol, dm_true_ao)
    validation_results['quadrupole_elec_error'] = np.linalg.norm(get_qudrupole_elec_vec(mlf_ftr.mol, dm_ml_ao) - get_qudrupole_elec_vec(mlf_ftr.mol, dm_true_ao))

    validation_results['dm_ml'] = dm_ml_ao
    

    # for i, target in enumerate(model_obj.targets):
    #     this_target = target.split('_')[0]
    #     diff_lo = veff_ml_lo[:,:,i] - np.linalg.multi_dot((C_ao_lo.T, mlf_true[this_target], C_ao_lo)) # in learning basis
    #     v_ml_ao = np.linalg.multi_dot((C_lo_ao.T, veff_ml_lo[:,:,i], C_lo_ao))
    #     diff_ao = v_ml_ao - mlf_true[this_target] # in ao basis 

    #     validation_results[f'{this_target}_mae_lo'] = np.mean(np.abs(diff_lo))
    #     validation_results[f'{this_target}_mae_ao'] = np.mean(np.abs(diff_ao))

    #     validation_results[f'etot_{this_target}_ml'] = np.einsum('ij,ji', v_ml_ao, dm_ml_ao)
    #     validation_results[f'etot_{this_target}_true'] = np.einsum('ij,ji', mlf_true[this_target], dm_true_ao)

    return validation_results


    
def validate_model_mol_core(chk_file, model_joblib_file):
    print(chk_file, flush = True)
    validation_results = {}

    veff_ml_lo, model_obj = predict_veff(chk_file, model_joblib_file, return_model = True)
    idx_fock = getattr(model_obj, 'idx_fock', 0)
    fock_ml_lo = veff_ml_lo[:,:,idx_fock]
    
    mlf_ftr = model_obj.pdata

    basis = model_obj.basis.split('_')[0]
    C_lo_ao = mlf_ftr[f'C_{basis}_ao'].copy()

    if 'delta' in model_obj.targets[0]:
        fock_ml_lo += model_obj.pdata[f'fock_{basis}']

    fock_ml_ao = np.linalg.multi_dot((C_lo_ao.T, fock_ml_lo, C_lo_ao)) # in learning basis
    

    mlf_true = lib.chkfile.load(chk_file, 'mlf')
    mlf_true['hcore+vj'] = mlf_true['vj'] + mlf_true['hcore']
    S_ao = mlf_true['ovlp']
    

    mo_ml, mo_coeff_ml = scipy.linalg.eigh(fock_ml_ao, S_ao)
    mo_true = mlf_true['mo_energy']

    mol = mlf_ftr.mol
    n_atoms = np.zeros(4, dtype = int) # C N O F
    for atom in mol._atom:
        symb = atom[0]
        if symb == 'C':
            n_atoms[0] += 1
        elif symb == 'N':
            n_atoms[1] += 1
        elif symb == 'O':
            n_atoms[2] += 1
        elif symb == 'F':
            n_atoms[3] += 1
    ncore = np.sum(n_atoms)
    assert ncore == len(mlf_ftr["inds_core"]), "Numbers of core orbitals do not match"
    mo_ml = mo_ml[:ncore]
    mo_true = mo_true[:ncore]
    mo_ml = np.flip(mo_ml)
    mo_true = np.flip(mo_true)
    sign_error = np.zeros(4)
    core_mae = np.zeros(4)
    # assuming 1s energy F < O < N < C and are the lowest mo
    for i in range(4):
        idx_start = np.sum(n_atoms[:i])
        idx_end = np.sum(n_atoms[:i+1])
        if idx_end > idx_start: # if there are atoms of this type
            core_mae[i] = np.mean(np.abs(mo_ml[idx_start:idx_end] - mo_true[idx_start:idx_end]))
            sign_error[i] = np.mean(mo_ml[idx_start:idx_end] - mo_true[idx_start:idx_end])

    validation_results['ncore_C'] = n_atoms[0]
    validation_results['ncore_N'] = n_atoms[1]
    validation_results['ncore_O'] = n_atoms[2]
    validation_results['ncore_F'] = n_atoms[3]
    validation_results['core_mae_C'] = core_mae[0]
    validation_results['core_mae_N'] = core_mae[1]
    validation_results['core_mae_O'] = core_mae[2]
    validation_results['core_mae_F'] = core_mae[3]
    validation_results['core_me_C'] = sign_error[0]
    validation_results['core_me_N'] = sign_error[1]
    validation_results['core_me_O'] = sign_error[2]
    validation_results['core_me_F'] = sign_error[3]

    validation_results["mo_energy_ml"] = mo_ml
    validation_results["mo_energy_true"] = mo_true

    return validation_results

def validate_model_mol_downfold_core(chk_file, model_joblib_file):
    print(chk_file, flush = True)
    validation_results = {}

    veff_ml_lo, model_obj = predict_veff(chk_file, model_joblib_file, return_model = True)
    idx_fock = getattr(model_obj, 'idx_fock', 0)
    fock_ml_lo = veff_ml_lo[:,:,idx_fock]
    
    mlf_ftr = model_obj.pdata

    basis = model_obj.basis.split('_')[0]
    C_ao_lo = mlf_ftr[f'C_ao_{basis}'].copy()
    C_lo_ao = mlf_ftr[f'C_{basis}_ao'].copy()

    if 'delta' in model_obj.targets[0]:
        fock_ml_lo += model_obj.pdata[f'fock_{basis}_downfold']

    fock_true_lo = mlf_ftr[f'fock_true_{basis}_downfold_exact']
    S_true_lo = mlf_ftr[f'S_{basis}_downfold_exact']
    fock_mae_lo = np.mean(np.abs(fock_true_lo - fock_ml_lo))
    validation_results['fock_lo_mae'] = fock_mae_lo

    mlf_true = lib.chkfile.load(chk_file, 'mlf')
    S_lo = mlf_ftr[f'S_{basis}_downfold']

    mo_ml, mo_coeff_ml_dwnfld = scipy.linalg.eigh(fock_ml_lo, S_true_lo)
    mo_true, mo_coeff_true_full = scipy.linalg.eigh(mlf_true['fock'], mlf_true['ovlp'])

    mol = mlf_ftr.mol
    n_atoms = np.zeros(4, dtype = int) # C N O F
    for atom in mol._atom:
        symb = atom[0]
        if symb == 'C':
            n_atoms[0] += 1
        elif symb == 'N':
            n_atoms[1] += 1
        elif symb == 'O':
            n_atoms[2] += 1
        elif symb == 'F':
            n_atoms[3] += 1
    ncore = np.sum(n_atoms)
    assert ncore == len(mlf_ftr["inds_core"]), "Numbers of core orbitals do not match"
    mo_ml = mo_ml[:ncore]
    mo_true = mo_true[:ncore]
    mo_ml = np.flip(mo_ml)
    mo_true = np.flip(mo_true)
    sign_error = np.zeros(4)
    core_mae = np.zeros(4)
    # assuming 1s energy F < O < N < C and are the lowest mo
    for i in range(4):
        idx_start = np.sum(n_atoms[:i])
        idx_end = np.sum(n_atoms[:i+1])
        if idx_end > idx_start: # if there are atoms of this type
            core_mae[i] = np.mean(np.abs(mo_ml[idx_start:idx_end] - mo_true[idx_start:idx_end]))
            sign_error[i] = np.mean(mo_ml[idx_start:idx_end] - mo_true[idx_start:idx_end])

    validation_results['ncore_C'] = n_atoms[0]
    validation_results['ncore_N'] = n_atoms[1]
    validation_results['ncore_O'] = n_atoms[2]
    validation_results['ncore_F'] = n_atoms[3]
    validation_results['core_mae_C'] = core_mae[0]
    validation_results['core_mae_N'] = core_mae[1]
    validation_results['core_mae_O'] = core_mae[2]
    validation_results['core_mae_F'] = core_mae[3]
    validation_results['core_me_C'] = sign_error[0]
    validation_results['core_me_N'] = sign_error[1]
    validation_results['core_me_O'] = sign_error[2]
    validation_results['core_me_F'] = sign_error[3]

    validation_results["mo_energy_ml"] = mo_ml
    validation_results["mo_energy_true"] = mo_true

    return validation_results
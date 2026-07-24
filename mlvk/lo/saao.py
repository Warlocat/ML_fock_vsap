import numpy as np
from mlvk.lo import make_basis
# from libdmet.basis_transform import make_basis
from pyscf.lo.iao import reference_mol

def get_custom_minao_dat(mol, custom_minao_path):

    if custom_minao_path is None:
        return None
    else:
        minao_val = custom_minao_path + '/minao_val.dat'
        symbols = list(set([a[0] for a in mol._atom if a[0].lower() != 'h']))

        minao_core = {}
        for atom in symbols:
            minao_core[atom] = f'{custom_minao_path}/minao_core.dat'

        return [minao_val, minao_core]

def get_saao(mol, dm, force_eigv_direction = False, return_idx = False):
    '''
    Symmetry-adapted atomic orbitals
    Ref: JCP 153, 124111 (2020)

    Return:
        C_ao_saao : (nao, nsaao) 2D array, each column corresponds to a SAAO
    '''
    # count AOs on each nl (n: 0,1,2,...; l: s, p, d...)
    nao_nl = []
    for i in range(mol.nbas):
        for k in range(mol.bas_nctr(i)):
            nao_nl.append(2 * mol.bas_angular(i) + 1)

    # AO index on each nl
    idx_tmp_start = 0
    idx_nl = []
    for i in range(len(nao_nl)):
        idx_tmp_end = idx_tmp_start + nao_nl[i]
        idx_nl.append([*range(idx_tmp_start, idx_tmp_end)])
        idx_tmp_start = idx_tmp_end

    # diagonalize each nl block of mean-field density matrix
    C_ao_saao = np.zeros_like(dm)
    for i in range(len(nao_nl)):
        ix = np.ix_(idx_nl[i], idx_nl[i])
        e_nl, y_nl = np.linalg.eigh(dm[ix])
        if force_eigv_direction:
            max_abs_idx = np.abs(y_nl).argmax(axis=0)
            for j in range(y_nl.shape[1]):
                y_nl[:,j] *= np.sign(y_nl[max_abs_idx[j], j])
        C_ao_saao[ix] = y_nl

    # exit()

    if return_idx:
        return C_ao_saao, idx_nl
    else:
        return C_ao_saao

def get_C_ao_iao(mf, minao, minao_val=None, minao_core=None, lowdin_orth = False, reorder_iao_first = True):
    '''
    Intrinsic atomic orbital + projected atomic orbital (IAO+PAO) basis
    Ref: JCTC 9, 4834-4843 (2013)

    Return:
        C_ao_iao : (nao, niao) 2D array, each column corresponds to a IAO or PAO
                    Note: the orbital order is the same as original AO basis
    '''

    if minao_val is None and minao_core is None:
        C_ao_iao = make_basis.get_C_ao_lo_iao_mol(mf, orth_core = lowdin_orth, orth_val = lowdin_orth, orth_virt = lowdin_orth, minao=minao, tol = 1e-6)
        

        mol = mf.mol
        pmol = reference_mol(mol, minao=minao)
        B1_labels = mol.ao_labels()
        B2_labels = pmol.ao_labels()
        iao_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B2_labels)]
        virt_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label not in B2_labels)]
        iao_virt_idx = iao_idx + virt_idx

        C_ao_iao_ordered = np.zeros_like(C_ao_iao)
        for i, iorb in enumerate(iao_virt_idx):
            C_ao_iao_ordered[:,iorb] = C_ao_iao[:,i]
    else:
        assert (minao_val is not None)
        assert (minao_core is not None)
        from libdmet.lo import iao
        mol = mf.mol
        print('Building pmol_core and pmol_val', flush = True)
        pmol_core, pmol_val = iao.build_pmol_core_val(mol, minao_core, minao_val)
        C_ao_iao = make_basis.get_C_ao_lo_iao_mol(mf, minao=minao, pmol_val=pmol_val,
                                                  pmol_core=pmol_core, orth_core = lowdin_orth, orth_val = lowdin_orth, orth_virt = lowdin_orth, tol = 1e-6)
        B1_labels = mol.ao_labels()
        B2_labels = pmol_core.ao_labels()
        B3_labels = pmol_val.ao_labels()
        core_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B2_labels)]
        val_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label in B3_labels)]
        virt_idx = [idx for idx, label in enumerate(B1_labels) \
                if (label not in B2_labels and label not in B3_labels)]
        core_val_virt_idx = core_idx + val_idx + virt_idx

        C_ao_iao_ordered = np.zeros_like(C_ao_iao)
        for i, iorb in enumerate(core_val_virt_idx):
            C_ao_iao_ordered[:,iorb] = C_ao_iao[:,i]

    if reorder_iao_first:
        return C_ao_iao_ordered
    else:
        return C_ao_iao

def lowdin_orth_wrapper(C_ao_myao, ovlp, inds_core = None):
    from libdmet.lo import iao
    if inds_core is not None:
        raise NotImplementedError('TODO: meta-lowdin like transformation for SAAOs')
        # inds_val = [i for i in range(ovlp.shape[0]) if i not in inds_core]
        # C_ao_lao_core = iao.vec_lowdin_k(C_ao_myao[:,inds_core], ovlp)
        # C_ao_lao_val = iao.vec_lowdin_k(C_ao_myao[:,inds_val], ovlp)
        # C_all = np.zeros_like(C_ao_myao)
        # C_all[:,inds_core] = C_ao_lao_core
        # C_all[:,inds_val] = C_ao_lao_val
        # return C_all
    else:
        return iao.vec_lowdin_k(C_ao_myao, ovlp)

def get_saaok(mol, fock, force_eigv_direction = False):
    '''
    Symmetry-adapted atomic orbitals
    Ref: JCP 153, 124111 (2020)
    Return:
        C_ao_saao : (nao, nsaao) 2D array, each column corresponds to a SAAO
    '''
    # count AOs on each nl (n: 0,1,2,...; l: s, p, d...)
    nao_nl = []
    for i in range(mol.nbas):
        for k in range(mol.bas_nctr(i)):
            nao_nl.append(2 * mol.bas_angular(i) + 1)
    # AO index on each nl
    idx_tmp_start = 0
    idx_nl = []
    for i in range(len(nao_nl)):
        idx_tmp_end = idx_tmp_start + nao_nl[i]
        idx_nl.append([*range(idx_tmp_start, idx_tmp_end)])
        idx_tmp_start = idx_tmp_end
    # diagonalize each nl block of mean-field density matrix
    C_ao_saao = np.zeros_like(fock)
    for k in range(fock.shape[0]):
        for i in range(len(nao_nl)):
            ix = np.ix_(idx_nl[i], idx_nl[i])
            e_nl, y_nl = np.linalg.eigh(fock[k][ix])
            if force_eigv_direction:
                slice_bool = np.sum(y_nl, axis = 0) < 0
                y_nl[:,slice_bool] = -y_nl[:,slice_bool]
            C_ao_saao[k][ix] = y_nl
    return C_ao_saao

def lowdin_orthk(mf_or_lattice, method='meta_lowdin', s = None):
    from libdmet.lo.lowdin import lowdin_k
    return lowdin_k(mf_or_lattice, method = method, s = s)
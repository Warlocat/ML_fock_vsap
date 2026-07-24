import scipy
import numpy as np

def get_C_ao_iao_vsap(mf, minao='minao',minao_val=None, minao_core=None, sap_type="dft", sap_basis = 'sapgrasplarge'):
    '''
    Get IAO+PAO basis from VSAP initial guess
    Diagonalize h+V_sap to get approx occupied orbitals, which gives approx IAOs
    Note: IAO+PAO orbital ordering is made the same as original AO
    '''
    from mlvk.lo.saao import get_C_ao_iao
    # Vsap initial guess
    mol = mf.mol
    assert sap_type.lower() in ["dft", "hf"] 
    if sap_type.lower() == "dft":
        Vsap = mf.get_vsap()
        T = mol.intor("int1e_kin")
        focksap = T + Vsap
    else:
        from pyscf.scf.hf import make_sap
        from mlvk.lib.ml_helper import get_sapbasis
        Vsap = make_sap(mol, sap_basis=get_sapbasis(mol, sap_basis))
        focksap = Vsap + mf.get_hcore()    
    S = mf.get_ovlp()
    mo_energy_vsap, mo_coeff_vsap = scipy.linalg.eigh(focksap, S)
    mo_occ_vsap = mf.get_occ(mo_energy_vsap)

    # get IAO from Vsap
    nocc = mol.nelectron // 2
    mf2 = mf.copy()
    mf2.mo_energy = mo_energy_vsap
    mf2.mo_coeff = mo_coeff_vsap
    mf2.mo_occ = mo_occ_vsap
    C_ao_iao = get_C_ao_iao(mf2, minao, minao_val=minao_val, minao_core=minao_core, lowdin_orth=True)

    return C_ao_iao

def get_C_ao_saiao_vsap(mf, minao='minao',minao_val=None, minao_core=None, full_return=True, sap_type="dft", sap_basis = 'sapgrasplarge'):
    '''
    Get symmetry-adapted IAO+PAO basis from VSAP initial guess
    Note: SAIAO orbital ordering is first all IAOs, then all PAOs
    '''
    # get IAO+PAO from vsap
    C_ao_iao = get_C_ao_iao_vsap(mf, minao=minao, minao_val=minao_val, minao_core=minao_core, sap_type=sap_type, sap_basis=sap_basis)

    # symmetrize using hcore
    hcore = mf.get_hcore()
    hcore_iao = C_ao_iao.T @ hcore @ C_ao_iao
    from mlvk.lo.saao import get_saao
    C_iao_saiao = get_saao(mf.mol, hcore_iao)
    assert(np.linalg.norm(np.dot(C_iao_saiao.T, C_iao_saiao)-np.eye(hcore.shape[0]))<1e-6)

    # AO to SAIAO coeff
    C_ao_saiao = np.dot(C_ao_iao, C_iao_saiao)

    # reorder to IAOs first, PAOs second
    C_ao_saiao_ordered = reorder_iaofirst(mf, C_ao_saiao, minao=minao, minao_val=minao_val, minao_core=minao_core)

    if full_return:
        from pyscf.lo.iao import reference_mol
        pmol = reference_mol(mf.mol, minao=minao)
        nminao = len(pmol.ao_labels())
        return C_ao_saiao_ordered, C_ao_saiao_ordered[:,:nminao]
    else:
        return C_ao_saiao_ordered

def reorder_iaofirst(mf, C_ao_iao, minao, minao_val=None, minao_core=None):
    '''
    Reorder IAO+PAO orbital ordering to be first all IAOs, then all PAOs
    i.e., [IAO_1, IAO_2, ..., PAO_1, PAO_2, ...]
    '''
    from pyscf.lo.iao import reference_mol
    if minao_val is None and minao_core is None:
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
            C_ao_iao_ordered[:,i] = C_ao_iao[:,iorb]
    else:
        assert (minao_val is not None)
        assert (minao_core is not None)
        from libdmet.lo import iao
        mol = mf.mol
        pmol_core, pmol_val = iao.build_pmol_core_val(mol, minao_core, minao_val)
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
            C_ao_iao_ordered[:,i] = C_ao_iao[:,iorb]

    return C_ao_iao_ordered

def reorder_ao_labels_iaofirst(mol, minao, minao_val=None, minao_core=None):
    '''
    Reorder ao_labels ordering to be first all IAOs, then all PAOs
    i.e., [IAO_1, IAO_2, ..., PAO_1, PAO_2, ...]
    '''
    from pyscf.lo.iao import reference_mol
    if minao_val is None and minao_core is None:
        pmol = reference_mol(mol, minao=minao)
        B1_labels = mol.ao_labels()
        B2_labels = pmol.ao_labels()
        iao_labels = [label for label in B1_labels if label in B2_labels]
        virt_labels = [label for label in B1_labels if label not in B2_labels]
        reordered_labels = iao_labels + virt_labels
    else:
        assert (minao_val is not None)
        assert (minao_core is not None)
        from libdmet.lo import iao
        pmol_core, pmol_val = iao.build_pmol_core_val(mol, minao_core, minao_val)
        B1_labels = mol.ao_labels()
        B2_labels = pmol_core.ao_labels()
        B3_labels = pmol_val.ao_labels()
        core_labels = [label for label in B1_labels if label in B2_labels]
        val_labels = [label for label in B1_labels if label in B3_labels]
        virt_labels = [label for label in B1_labels if (label not in B2_labels and label not in B3_labels)]
        reordered_labels = core_labels + val_labels + virt_labels

    return reordered_labels

def get_C_ao_salowdin(mf, proj_basis='minao', full_return=True):
    '''
    Get symmetry-adapted Lowdin basis = proj_basis + remaining high virtuals
    The order of orbitals is first all small proj_basis, then all remaining high virtuals
    i.e., [proj_basis_1, proj_basis_2, ..., highvir_1, highvir_2, ...]
    '''
    if proj_basis == mf.mol.basis or proj_basis is None:
        # NOTE: the order of normal symmetry adapted local orbitals is
        # the same as the original AOs
        from pyscf import lo
        C_ao_lowdin = lo.orth_ao(mf.mol, 'meta_lowdin')
        hcore = mf.get_hcore()
        hcore_lo = C_ao_lowdin.T @ hcore @ C_ao_lowdin
        from mlvk.lo.saao import get_saao
        C_lo_salo = get_saao(mf.mol, hcore_lo)
        assert(np.linalg.norm(np.dot(C_lo_salo.T, C_lo_salo)-np.eye(hcore_lo.shape[0]))<1e-6)
        C_ao_lo = C_ao_lowdin @ C_lo_salo
        return C_ao_lo

    from pyscf import gto
    # build new mol object with proj_basis
    mol = mf.mol
    mol2 = gto.M(atom=mol.atom,
             unit=mol.unit,
             charge=mol.charge,
             spin=mol.spin,
             basis=proj_basis)

    # calculate overlap matrices
    S_lg_sm = gto.intor_cross('int1e_ovlp', mol, mol2)
    S_lg = mol.intor('int1e_ovlp')

    # find remaining high virtual basis
    B1_labels = mol.ao_labels()
    B2_labels = mol2.ao_labels()
    virt_idx = [idx for idx, label in enumerate(B1_labels)
                if (not label in B2_labels)]
    nB1 = len(B1_labels)
    nB2 = len(B2_labels)
    nvirt = len(virt_idx)
    assert nB2 + nvirt == nB1

    C_aolg_aosm = np.linalg.solve(S_lg, S_lg_sm)
    CSC = np.dot(C_aolg_aosm.T, S_lg).dot(C_aolg_aosm)
    CSC_inv = np.linalg.inv(CSC)
    # construct the projector
    P = C_aolg_aosm @ CSC_inv @ C_aolg_aosm.T @ S_lg
    C_aolg_aovirt = (np.eye(nB1) - P)[:, virt_idx]

    # full AO basis : minao + high virtuals
    C_aolg_full = np.hstack((C_aolg_aosm, C_aolg_aovirt))

    # orthogonal lowdin local orbitals
    from pyscf.lo import orth
    S_full = C_aolg_full.T @ S_lg @ C_aolg_full
    X = orth.lowdin(S_full)
    C_aolg_lo = C_aolg_full @ X

    # order lowdin orbitals to be same as original AOs
    minao_idx = [idx for idx, label in enumerate(B1_labels) \
            if (label in B2_labels)]
    minao_virt_idx = minao_idx + virt_idx

    C_ao_lo_ordered = np.zeros_like(C_aolg_lo)
    for i, iorb in enumerate(minao_virt_idx):
        C_ao_lo_ordered[:,iorb] = C_aolg_lo[:,i]

    # symmetry adaption
    hcore = mf.get_hcore()
    hcore_lo = C_ao_lo_ordered.T @ hcore @ C_ao_lo_ordered
    from mlvk.lo.saao import get_saao
    C_lo_salo = get_saao(mol, hcore_lo)
    assert(np.linalg.norm(np.dot(C_lo_salo.T, C_lo_salo)-np.eye(hcore_lo.shape[0]))<1e-6)

    C_ao_lo_ordered = np.dot(C_ao_lo_ordered, C_lo_salo)
    C_ao_lo = reorder_iaofirst(mf, C_ao_lo_ordered, minao=proj_basis)
    C_ao_minlo = C_ao_lo[:,:nB2]

    if full_return:
        return C_ao_lo, C_ao_minlo
    else:
        return C_ao_lo

import os
import numpy as np
import scipy
from scipy.optimize import newton
from pyscf.data import elements
from pyscf.lib import logger
from pyscf.scf.atom_hf import AtomHF1e, AtomSphAverageRHF

def fock_downfold(mf, omega, proj_basis='minao', lo_method='iao', local_frag=False,
                  idx_frag=None, delta=0.001, fermi_adjust=False, full_return=True,
                  sap_type="dft", sap_basis = 'sapgrasplarge'):
    '''
    Use lowdin partitioning to derive an effective Fock matrix (Hamiltonian)
    Heff(w) = H_AA + hyb(w), where hyb(w) = H_AR (w-H_RR)^-1 H_RA
    Subspace can be minimal basis of all atoms, or all local orbitals on a fragment

    Parameters
    ----------
    mf : mean-field object
    omega : complex 1D array
        Frequencies on which Heff is calculated
    proj_basis : string
        Minimal basis to be projected onto
    lo_method : string
        Symmetry-adapated localization methods: iao or lowdin
    local_frag: bool
        Whether the downfolding is for a local fragment
    idx_frag : list
        Orbital indices that belong to the downfolded fragment,
        only used when local_frag==True
    delta : float
        Broadening for calculating Heff
    fermi_adjust : bool
        Whether to shift all mo_energy wrt Fermi level
    full_return : bool
        Whether to return all essential quantities

    Returns
    -------
    h_AA_w : complex 3D array
        Downfolded dynamical Hamiltonian, (n_eff, n_eff, nw)
    h_AA : double 2D array
        Bare static Hamiltonian, (neff, neff)
    hyb : complex 3D array
        Hybridization function
    C_ao_lo : double 2D array
        AO to all LO coefficients
    C_ao_minao : double 2D array
        AO to LO (only minimal basis or local fragment) coefficients
    '''
    # get symmetry-adpated orthogonal local basis
    if lo_method == 'iao':
        # NOTE: order: first all IAOs, then all PAOs
        from mlvk.lo.saiao_vsap import get_C_ao_saiao_vsap
        C_ao_lo, C_ao_minao = get_C_ao_saiao_vsap(mf, minao=proj_basis, sap_type=sap_type, sap_basis=sap_basis)
    elif lo_method == 'lowdin':
        # NOTE: order: first all sa-lowdin on minao, then all high virtual lowdin (without sa)
        from mlvk.lo.saiao_vsap import get_C_ao_salowdin
        C_ao_lo, C_ao_minao = get_C_ao_salowdin(mf, proj_basis=proj_basis)
    else:
        raise NotImplementedError

    if local_frag:
        # derive effective Hamiltonian for a local fragment (all local orbitals)
        # TODO: possible to use normal SA-Lowdin (no minao projection)
        assert (lo_method == 'iao' or lo_method == 'lowdin')
        assert idx_frag is not None
        C_ao_minao = C_ao_lo[:,idx_frag]
        idx_all = list(range(C_ao_minao.shape[0]))
        idx_rest = [x for x in idx_all if (x not in idx_frag)]
        C_ao_rest = C_ao_lo[:,idx_rest]
        C_ao_lo = np.hstack([C_ao_minao, C_ao_rest])

    # Lowdin partitioning
    if fermi_adjust:
        print ('!! Note: The effective Hamiltonian is shifted relative to the Fermi level !!')
        ef = get_ef(mf)
        fock_mo = np.diag(mf.mo_energy - ef)
    else:
        fock_mo = np.diag(mf.mo_energy)
    S = mf.get_ovlp()
    fock = S @ mf.mo_coeff @ fock_mo @ mf.mo_coeff.T @ S
    h_new = C_ao_lo.T @ fock @ C_ao_lo

    nB1, nB2 = C_ao_minao.shape
    h_AA = h_new[:nB2,:nB2]
    h_RR = h_new[nB2:,nB2:]
    h_AR = h_new[:nB2,nB2:]

    # Pre-diagonalize h_RR once (assume Hermitian)
    e_RR, U_RR = scipy.linalg.eigh(h_RR)
    A = h_AR @ U_RR              # (nB2, nR)
    B = U_RR.conj().T @ h_AR.T     # (nR, nB2)

    nw = len(omega)
    h_AA_w = np.zeros((nB2, nB2, nw),dtype=np.complex128)
    hyb = np.zeros((nB2, nB2, nw),dtype=np.complex128)
    for iw in range(nw):
        denom = 1.0 / (omega[iw] + 1j * delta - e_RR)
        hyb[:, :, iw] = A * denom[None, :] @ B
        h_AA_w[:, :, iw] = h_AA + hyb[:, :, iw]

        # more expensive way to calculate: h_eff(w) = h_AA + h_AR (w-h_RR)^-1 h_RA
        #h_RR_inv = np.linalg.inv((omega[iw] + 1j*delta)*np.eye(nB1-nB2) - h_RR)
        #hyb[:,:,iw] = np.dot(h_AR, h_RR_inv).dot(h_AR.T)
        #h_AA_w[:,:,iw] = h_AA + hyb[:,:,iw]

    if full_return:
        return h_AA_w, h_AA, hyb, C_ao_lo, C_ao_minao
    else:
        return h_AA_w

def fock_downfold_x(mf, proj_basis='minao', lo_method='iao', local_frag=False, idx_mo=None,
                    idx_frag=None, fermi_adjust=False, full_return=True,
                    exclude_core=False, pmol_core=None,
                    sap_type="dft", sap_basis = 'sapgrasplarge'):
    '''
    Use an freq-free method to derive an effective Fock matrix (Hamiltonian)
    This is an anology to the exact two-component (X2C) method in relativistic quantum chemistry.
    Similar terminology (for X and R matrices) is used here.

    For orthogonal basis partitioned as A (subspace) and R (rest):
        | H_AA H_AR | | C_A | = E  | C_A |
        | H_RA H_RR | | C_R |      | C_R |
    Solve the X matrix s.t. C_R = X C_A, then eliminate C_R.
    We have the following Hermitian Hamiltonian: 
        H' = H_AA + H_AR X + X^dagger H_RA + X^dagger H_RR X
        S' = I + X^dagger X
    and
        H' C_A = E S' C_A
    Following the X2C procedure, we renormalize the effective Hamiltonian so that
    we have normalized eigenvectors:
        R C_eff = C_A, with R = S'^(-1/2)
        H_eff = R^dagger H' R
    and finally
        H_eff C_eff = E C_eff

    For non-orthogonal basis, the definition of R matrix is more complicated,
    please see X2C literature for details.

    Parameters
    ----------
    mf : mean-field object
    proj_basis : string
        Minimal basis to be projected onto
    lo_method : string
        Symmetry-adapated localization methods: iao or lowdin
    local_frag: bool
        Whether the downfolding is for a local fragment
    idx_frag : list
        Orbital indices that belong to the downfolded fragment,
        only used when local_frag==True
    fermi_adjust : bool
        Whether to shift all mo_energy wrt Fermi level
    full_return : bool
        Whether to return all essential quantities

    Returns
    -------
    h_AA_w : complex 3D array
        Downfolded dynamical Hamiltonian, (n_eff, n_eff, nw)
    h_AA : double 2D array
        Bare static Hamiltonian, (neff, neff)
    hyb : complex 3D array
        Hybridization function
    C_ao_lo : double 2D array
        AO to all LO coefficients
    C_ao_minao : double 2D array
        AO to LO (only minimal basis or local fragment) coefficients
    '''
    # get symmetry-adpated orthogonal local basis
    if lo_method == 'iao':
        # NOTE: order: first all IAOs, then all PAOs
        from mlvk.lo.saiao_vsap import get_C_ao_saiao_vsap
        C_ao_lo, C_ao_minao = get_C_ao_saiao_vsap(mf, minao=proj_basis, sap_type=sap_type, sap_basis=sap_basis)
    elif lo_method == 'lowdin':
        # NOTE: order: first all sa-lowdin on minao, then all high virtual lowdin (without sa)
        from mlvk.lo.saiao_vsap import get_C_ao_salowdin
        C_ao_lo, C_ao_minao = get_C_ao_salowdin(mf, proj_basis=proj_basis)
    else:
        raise NotImplementedError

    if local_frag:
        # derive effective Hamiltonian for a local fragment (all local orbitals)
        # TODO: possible to use normal SA-Lowdin (no minao projection)
        raise NotImplementedError
        

    # Lowdin partitioning
    if fermi_adjust:
        print ('!! Note: The effective Hamiltonian is shifted relative to the Fermi level !!')
        ef = get_ef(mf)
        fock_mo = np.diag(mf.mo_energy - ef)
    else:
        fock_mo = np.diag(mf.mo_energy)
    

    if exclude_core:
        assert pmol_core is not None
        # reorder the IAO columns so that core comes after the valence
        # as if the core orbitals are treated as part of the "rest" space R
        from pyscf.lo.iao import reference_mol
        pmol = reference_mol(mol, minao=proj_basis)
        core_idx = [idx for idx, label in enumerate(pmol.ao_labels()) if (label in pmol_core.ao_labels())]
        other_idx = [idx for idx in range(C_ao_lo.shape[1]) if idx not in core_idx]
        C_ao_lo_core = C_ao_lo[:, core_idx]
        C_ao_lo_other = C_ao_lo[:, other_idx]
        C_ao_lo = np.hstack([C_ao_lo_other, C_ao_lo_core])
        other_idx = [idx for idx in range(C_ao_minao.shape[1]) if idx not in core_idx]
        C_ao_minao = C_ao_minao[:, other_idx]


    S = mf.get_ovlp()
    fock_ao = S @ mf.mo_coeff @ fock_mo @ mf.mo_coeff.T @ S
    h_new = C_ao_lo.T @ fock_ao @ C_ao_lo
    nB1, nB2 = C_ao_minao.shape

    # Pre-diagonalize h_RR once (assume Hermitian)
    if exclude_core:
        x_mat, idx_mo = get_X_matrix(h_new, nB2, idx_ao = other_idx)
    else:
        x_mat, idx_mo = get_X_matrix(h_new, nB2, idx_mo=idx_mo)
    h_prime = get_h_prime(h_new, x_mat)
    s_prime = np.eye(nB2) + x_mat.T @ x_mat
    from pyscf.x2c import x2c
    r_mat = x2c._invsqrt(s_prime)
    h_eff = r_mat.T @ h_prime @ r_mat

    if full_return:
        return h_eff, x_mat, r_mat, C_ao_lo, C_ao_minao, idx_mo
    else:
        return h_eff
    
def get_X_matrix(H, n, idx_mo = None, idx_ao = None):
    '''
    Get X matrix from Hamiltonian and selected MO
    '''
    _, mo_coeff = np.linalg.eigh(H)

    if idx_mo is not None:
        assert len(idx_mo) == n
    else:
        if idx_ao is None:
            idx_ao = list(range(n))
        idx_mo = get_optimal_mo_idx(mo_coeff, idx_ao)

    mo_coeff_A = mo_coeff[:n,:][:,idx_mo]
    mo_coeff_R = mo_coeff[n:,:][:,idx_mo]

    # X_mat = scipy.linalg.solve(mo_coeff_A, mo_coeff_R.T, transposed = True).T

    # p, res, rnk, s = scipy.linalg.lstsq(mo_coeff_A.T, mo_coeff_R.T)
    # X_mat = p.T

    # X_mat = mo_coeff_R @ np.linalg.pinv(mo_coeff_A)

    X_mat = mo_coeff_R @ mo_coeff_A.T @ np.linalg.pinv(mo_coeff_A @ mo_coeff_A.T)
    

    X_check = np.linalg.norm(X_mat @ mo_coeff_A - mo_coeff_R)
    if X_check > 1e-5:
        print("X performance check: ", X_check)
        print("X norm: ", np.linalg.norm(X_mat))
        X_mat = mo_coeff_R @ np.linalg.pinv(mo_coeff_A,rcond=1e-4)
        X_check = np.linalg.norm(X_mat @ mo_coeff_A - mo_coeff_R)
        print("X performance check after pinv: ", X_check)
    return X_mat, idx_mo
    
def get_h_prime(H, x_mat):
    '''
    Get H' = H_AA + H_AR X + X^dagger H_RA + X^dagger H_RR X
    '''
    nB2 = x_mat.shape[1]
    h_AA = H[:nB2,:nB2]
    h_RR = H[nB2:,nB2:]
    h_AR = H[:nB2,nB2:]
    h_prime = h_AA + h_AR @ x_mat + x_mat.T @ h_AR.T + x_mat.T @ h_RR @ x_mat
    return h_prime
    
def get_downfold_matrix(h_mat, x_mat):
    '''
    Get H_eff = R^dagger H' R
    this function assumes orthonormal basis
    '''
    nB2 = x_mat.shape[1]
    h_prime = get_h_prime(h_mat, x_mat)
    s_prime = np.eye(nB2) + x_mat.T @ x_mat
    from pyscf.x2c import x2c
    r_mat = x2c._invsqrt(s_prime)
    return r_mat.T @ h_prime @ r_mat

def get_downfold_matrix_extra(h_mat, x_mat):
    '''
    Get H_eff = R^dagger H' R
    this function assumes orthonormal basis
    '''
    nB2 = x_mat.shape[1]
    h_AA = h_mat[:nB2,:nB2]
    h_RR = h_mat[nB2:,nB2:]
    h_AR = h_mat[:nB2,nB2:]
    h_hyb = h_AR @ x_mat + x_mat.T @ h_AR.T + x_mat.T @ h_RR @ x_mat
    s_prime = np.eye(nB2) + x_mat.T @ x_mat
    from pyscf.x2c import x2c
    r_mat = x2c._invsqrt(s_prime)
    return h_AA, h_hyb, r_mat.T @ h_AA @ r_mat, r_mat.T @ h_hyb @ r_mat

def get_optimal_mo_idx(mo_coeff, ao_idx, ovlp_b2_b1 = None):
    if ovlp_b2_b1 is None:
        ovlp_b2_b1 = np.zeros((len(ao_idx), mo_coeff.shape[0]))
        for i in range(len(ao_idx)):
            ovlp_b2_b1[i,i] = 1.0
    ovlp_ao_mo = ovlp_b2_b1 @ mo_coeff
    n_b2 = len(ao_idx)
    norm = np.linalg.norm(ovlp_ao_mo, axis=0)
    print(norm)
    idx_opt = np.argsort(norm)[::-1][:n_b2]
    idx_opt = sorted(idx_opt)
    print("Optimal MO indices: ", idx_opt)
    return idx_opt


def get_atm_nrhf(mol, atomic_configuration=elements.NRSRHF_CONFIGURATION):
    elements = {a[0] for a in mol._atom}
    logger.info(mol, 'Spherically averaged atomic HF for %s', elements)

    atm_template = mol.copy(deep=False)
    atm_template.charge = 0
    atm_template.enuc = 0
    atm_template.symmetry = False  # TODO: enable SO3 symmetry here
    atm_template.atom = atm_template._atom = []
    atm_template.cart = False  # AtomSphAverageRHF does not support cartesian basis

    atm_scf_result = {}
    for ia, a in enumerate(mol._atom):
        element = a[0]
        if element in atm_scf_result:
            continue

        atm = atm_template
        atm._atom = [a]
        atm._atm = mol._atm[ia:ia+1]
        atm._bas = mol._bas[mol._bas[:,0] == ia].copy()
        atm._ecpbas = mol._ecpbas[mol._ecpbas[:,0] == ia].copy()
        # Point to the only atom
        atm._bas[:,0] = 0
        atm._ecpbas[:,0] = 0
        if element in mol._pseudo:
            atm._pseudo = {element: mol._pseudo.get(element)}
        atm.spin = atm.nelectron % 2

        nao = atm.nao
        # nao == 0 for the case that no basis was assigned to an atom
        if nao == 0 or atm.nelectron == 0:  # GHOST
            mo_occ = mo_energy = np.zeros(nao)
            mo_coeff = np.zeros((nao,nao))
            atm_scf_result[element] = (0, mo_energy, mo_coeff, mo_occ)
        elif atm._pseudo:
            from pyscf.scf import atom_hf_pp
            atm.a = None
            if atm.nelectron == 1:
                atm_hf = atom_hf_pp.AtomHF1ePP(atm)
            else:
                atm_hf = atom_hf_pp.AtomSCFPP(atm)
                atm_hf.atomic_configuration = atomic_configuration

            atm_hf.verbose = mol.verbose
            atm_hf.run()
            atm_scf_result[element] = (atm_hf.e_tot, atm_hf.mo_energy,
                                       atm_hf.mo_coeff, atm_hf.mo_occ)
        else:
            if atm.nelectron == 1:
                atm_hf = AtomHF1e(atm)
            else:
                atm_hf = AtomSphAverageRHF(atm)
                atm_hf.atomic_configuration = atomic_configuration

            atm_hf.verbose = mol.verbose
            atm_hf.run()
            atm_scf_result[element] = (atm_hf.e_tot, atm_hf.mo_energy,
                                       atm_hf.mo_coeff, atm_hf.mo_occ,
                                       atm_hf.get_ovlp(), atm_hf.get_hcore())
    return atm_scf_result



def solve_heff_QP(h_static, hyb, iomega, roots=None, eta=1e-4, etol=1e-4, max_cycle=20):
    '''
    Self-consistently solve Heff(E) C = E C through analytic continuation
    '''
    from fcdmft.ac.pade import pade_thiele
    # solve QP equations in diagonal-hyb approximation (MO basis)
    nao, nao, nw = hyb.shape
    mo_energy, mo_coeff = scipy.linalg.eigh(h_static)
    hyb_mo = np.zeros((nao,nao,nw),dtype=np.complex128)
    for iw in range(nw):
        hyb_mo[:,:,iw] = mo_coeff.T @ hyb[:,:,iw] @ mo_coeff
    h_static_mo = mo_coeff.T @ h_static @ mo_coeff

    coeff, omega_fit = my_AC_pade_thiele_diag(hyb_mo, iomega)

    mo_energy_new = np.zeros_like(mo_energy)
    for p in range(nao):
        def quasiparticle(omega):
            sigmaR = pade_thiele(omega + 1j*eta, omega_fit, coeff[:, p]).real
            return omega - mo_energy[p] - sigmaR
        try:
            mo_energy_new[p] = newton(quasiparticle, mo_energy[p], tol=1e-6, maxiter=50)
        except RuntimeError:
            print("QPE for orbital=%d not converged!" % p)

    # use approx QP energies to solve Heff(E) C = E C self-consistently
    coeff, omega_fit = my_AC_pade_thiele_full(hyb_mo, iomega)

    mo_energy_old = mo_energy.copy()
    mo_coeff_new = mo_coeff.copy()
    if roots is None:
        roots = range(nao)

    # TODO: DIIS to accelerate convergence
    for p in roots:
        cycle = 0
        while abs(mo_energy_new[p]-mo_energy_old[p]) > etol or cycle == 0:
            Heff = np.zeros((nao, nao))
            for i in range(nao):
                for j in range(nao):
                    Heff[i,j] = pade_thiele(mo_energy_new[p]+ 1j*eta, omega_fit, coeff[:, i, j]).real + h_static_mo[i,j]
            Heff = 0.5 * (Heff + Heff.T)
            e, c = scipy.linalg.eigh(Heff)

            mo_energy_old[p] = mo_energy_new[p]
            mo_energy_new[p] = e[p]
            mo_coeff_new[:,p] = np.dot(mo_coeff, c[:,p])
            cycle = cycle + 1
            if cycle > max_cycle:
                print ('orbital %d self-consistent solution not converged!' % p)
                break

    return mo_energy_new, mo_coeff_new

def solve_Heff_scan(Heff, omega, nroots=None):
    '''
    Scan Heff(w)C = wC to find lowest roots of Heff

    Parameters
    ----------
    Heff : complex 3D array
        Downfolded dynamical Hamiltonian, (n_eff, n_eff, nw)
    omega : double 1D array
        Excitation frequencies on which Heff is calculated
    nroots : int
        Number of roots to scan

    Returns
    -------
    Print out the first nroots scan over a range of energy points
    '''
    if nroots is None:
        nroots = Heff.shape[0]

    nw = len(omega)
    e_nroots = np.zeros((nroots, nw))
    for iw in range(nw):
        e,c = scipy.linalg.eigh(Heff[:,:,iw].real)
        e_nroots[:,iw] = e[:nroots]

    for i in range(nroots):
        print(f"Scan Root {i+1} : Energy(eV)  Residue")
        for iw in range(nw):
            print("{:.6f} {:.6f}".format(omega[iw], e_nroots[i,iw] - omega[iw]))
        print ('----------')

def my_AC_pade_thiele_full(sigma, omega, step_ratio=2.0/3.0):
    """
    Analytic continuation to real axis using a Pade approximation

    Args:
        sigma : 3D array (norbs, norbs, nomega)
        omega : 1D array (nomega)

    Returns:
        coeff: 3D array (ncoeff, norbs, norbs)
        omega : 1D array (ncoeff)
    """
    from fcdmft.ac.pade import thiele
    norbs, norbs, nw = sigma.shape
    npade = nw // 2 # take even number of points.
    coeff = np.zeros((npade*2, norbs, norbs),dtype=np.complex128)
    for p in range(norbs):
        for q in range(norbs):
            coeff[:,p,q] = thiele(sigma[p,q,:npade*2], omega[:npade*2])
    return coeff, omega[:npade*2]

def my_AC_pade_thiele_diag(sigma, omega, step_ratio=2.0/3.0):
    """
    Analytic continuation to real axis using a Pade approximation

    Args:
        sigma : 3D array (norbs, norbs, nomega)
        omega : 1D array (nomega)

    Returns:
        coeff: 2D array (ncoeff, norbs)
        omega : 1D array (ncoeff)
    """
    from fcdmft.ac.pade import thiele
    norbs, norbs, nw = sigma.shape
    npade = nw // 2 # take even number of points.
    coeff = np.zeros((npade*2, norbs),dtype=np.complex128)
    for p in range(norbs):
        coeff[:,p] = thiele(sigma[p,p,:npade*2], omega[:npade*2])
    return coeff, omega[:npade*2]

def get_iomega():
    '''
    Get imaginary frequency grid for performing AC
    '''
    from fcdmft.ac.grids import _get_scaled_legendre_roots
    from fcdmft.ac.pade import _get_ac_idx
    freqs, wts = _get_scaled_legendre_roots(nw=100)
    freqs = np.concatenate([[0],freqs])
    nw_sigma = sum(freqs < 5.0) + 1
    freqs = freqs[:nw_sigma]
    idx = _get_ac_idx(len(freqs), idx_start=1)
    iomega = 1j * freqs[idx]
    return iomega

def solve_Heff_AC(omega, ac_coeff, omega_fit, eta=0.001, nroots=None):
    '''
    First construct Heff(w) from Pade coefficients,
    then scan Heff(w)C = wC to find lowest roots of Heff

    Parameters
    ----------
    omega : double 1D array
        Real excitation frequencies on which Heff is calculated
    coeff: complex 2D array
        Pade AC cofficients, (ncoeff, norbs, norbs)
    omega_fit : complex 1D array
        Imaginary frequencies for Pade AC, (ncoeff)
    nroots : int
        Number of roots to scan
    eta : double
        Broadening factor, default: 0.001 a.u.

    Returns
    -------
    Print out the first nroots scan over a range of energy points
    Heff : complex 3D array
        Downfolded dynamical TDA Hamiltonian, (nocc_act*nvir_act, nocc_act*nvir_act, nw)
    '''
    from fcdmft.ac.pade import pade_thiele
    nw = len(omega)
    npair = ac_coeff.shape[-1]

    # get Heff on the real axis
    Heff = np.zeros(shape=[npair, npair, nw], dtype=np.complex128)
    for i in range(npair):
        for j in range(npair):
            Heff[i,j] = pade_thiele(omega + 1j * eta, omega_fit, ac_coeff[:, i, j])

    if nroots is None:
        nroots = Heff.shape[0]

    nw = len(omega)
    e_nroots = np.zeros((nroots, nw))
    for iw in range(nw):
        e,c = scipy.linalg.eigh(Heff[:,:,iw].real)
        e_nroots[:,iw] = e[:nroots]

    for i in range(nroots):
        print(f"Scan Root {i+1} : Energy(eV)  Residue")
        for iw in range(nw):
            print("{:.6f} {:.6f}".format(omega[iw], e_nroots[i,iw] - omega[iw]))
        print ('----------')

    return Heff

def get_ef(mf, mo_energy=None):
    """Get Fermi level.
    For gapped systems, Fermi level is computed as the average between HOMO and LUMO.
    For metallic systems, Fermi level is optmized according to mo_energy.

    Parameters
    ----------
    mo_energy : double 1d array, optional
        orbital energy, by default None

    Returns
    -------
    ef : double
        Fermi level
    """
    if mo_energy is None:
        mo_energy = mf.mo_energy

    if hasattr(mf, 'sigma'):
        f_occ = _fermi_smearing_occ if mf.smearing_method.lower() == 'fermi' else _gaussian_smearing_occ
        ef = _smearing_optimize(f_occ, mo_energy, mf.mol.nelectron / 2, mf.sigma)[0]
    else:
        # working with full space mo_energy and nocc here
        nocc = mf.mol.nelectron // 2
        ef = (mo_energy[nocc - 1] + mo_energy[nocc]) * 0.5
    return ef

def get_gf_fromAC(h_static, hyb, iomega, freqs, eta):
    '''
    Calculate real-axis Green's function from h_static and imag-freq hyb

    Parameters
    ----------
    h_static : double 2D array
        Bare Fock matrix on the subspace
    hyb : complex 3D array
        Hybridization function
    iomega : complex 1D array
        Imaginary frequencies on which hyb is calculated
    freqs : double 1D array
        Real frequencies on which GF is calculated
    eta : double
        Broadening factor

    Returns
    -------
    gf : complex 3D array
        Mean-field Green's function on real axis
    '''
    from fcdmft.ac.pade import pade_thiele
    nao, nao, nw_imag = hyb.shape
    coeff, omega_fit = my_AC_pade_thiele_full(hyb, iomega)

    nw_real = len(freqs)
    hyb_real = np.zeros(shape=[nao, nao, nw_real], dtype=np.complex128)
    for i in range(nao):
        for j in range(nao):
            hyb_real[i,j] = pade_thiele(freqs + 1j * eta, omega_fit, coeff[:, i, j])

    gf = np.zeros((nao, nao, nw_real), dtype=np.complex128)
    for iw in range(nw_real):
        gf[:,:,iw] = np.linalg.inv((freqs[iw] + 1j*eta)*np.eye(nao) - h_static - hyb_real[:,:,iw])

    return gf

def orthogonalize_mo(mo_coeff, overlap=None):
    if overlap is None:
        m_metric = np.dot(mo_coeff.T, mo_coeff)
    else:
        m_metric = np.dot(np.dot(mo_coeff.T, overlap), mo_coeff)
    evals, evecs = scipy.linalg.eigh(m_metric)
    evals = np.abs(evals)
    evals_inv_sqrt = 1.0 / np.sqrt(evals)
    x_mat = np.dot(evecs, np.dot(np.diag(evals_inv_sqrt), evecs.T))
    mo_orth = np.dot(mo_coeff, x_mat)

    return mo_orth

def test_dip_moment(mf, C_ao_iao, mo_coeff_sm):
    from pyscf.scf.hf import dip_moment
    dm = mf.make_rdm1()
    dip = dip_moment(mf.mol, dm)

    C_ao_mosm = np.dot(C_ao_iao, mo_coeff_sm)
    nocc = mf.mol.nelectron // 2
    dm_eff = 2. * np.dot(C_ao_mosm[:,:nocc], C_ao_mosm[:,:nocc].T)
    dip_eff = dip_moment(mf.mol, dm_eff)

    return dip, dip_eff

def get_dip_moment_x(mol, dm, x_x2c, r_x2c, C_ao_lo, picture_change=True, nuc=True):
    '''
    Calculate dipole moment in the effective X2C Hamiltonian framework
    '''
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    origin = np.zeros(3)
    if nuc:
        nucl_dip = np.einsum('i,ix->x', charges, coords - origin[None, :])
    else:
        nucl_dip = np.zeros(3)
    with mol.with_common_orig(origin):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    for i in range(3):
        ao_dip[i] = C_ao_lo.T @ ao_dip[i] @ C_ao_lo
    if not picture_change:
        n_s, n_l = x_x2c.shape
        ao_dip_x2c = ao_dip[:, :n_l, :n_l]
    else:
        ao_dip_x2c = []
        for i in range(3):
            ao_dip_i = get_h_prime(ao_dip[i], x_x2c)
            ao_dip_i = r_x2c.T @ ao_dip_i @ r_x2c
            ao_dip_x2c.append(ao_dip_i)
        ao_dip_x2c = np.array(ao_dip_x2c)
    el_dip = np.einsum('xij,ji->x', ao_dip_x2c, dm).real
    mol_dip = nucl_dip - el_dip
    from pyscf.data import nist
    mol_dip *= nist.AU2DEBYE
    from pyscf.lib import logger
    log = logger.new_logger(mf, mol.verbose)
    log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)

    return mol_dip

if __name__ == '__main__':
    from pyscf import gto, scf, dft
    mol = gto.Mole()
    mol.verbose = 4
    # mol.atom = '''
    # O       -0.58889       -0.42918        0.00000
    # H        0.56496       -0.41602        0.00000
    # H       -0.91222        0.30740        0.54205
    # '''
    mol.atom = '''
C      3.4932470000   -0.1059880000    0.2791310000
C      2.2443260000    0.4558670000   -0.4109130000
C      1.0549330000   -0.4656040000   -0.2490870000
N      0.3089620000   -0.3921250000    1.0171970000
C     -0.3838330000   -0.0558720000   -0.2428430000
C     -0.8123930000    1.3965910000   -0.3862780000
C     -2.1039580000    1.7312820000    0.3731660000
C     -1.3926320000   -1.0942830000   -0.7361890000
O     -2.2555540000   -1.5738270000    0.2910950000
H      3.3097840000   -0.2416730000    1.3494350000
H      3.7754710000   -1.0787870000   -0.1392140000
H      4.3480000000    0.5670800000    0.1622580000
H      2.0121680000    1.4387300000    0.0086840000
H      2.4419390000    0.6025100000   -1.4809350000
H      1.2375420000   -1.4661210000   -0.6459020000
H      0.0034950000   -1.3118050000    1.3284220000
H     -0.0135870000    2.0433800000   -0.0158530000
H     -0.9398470000    1.6288010000   -1.4520210000
H     -2.3293190000    2.8002940000    0.3130100000
H     -2.9731270000    1.2041700000   -0.0393540000
H     -2.0053380000    1.4692560000    1.4318130000
H     -1.9693100000   -0.6830240000   -1.5786930000
H     -0.8748280000   -1.9862820000   -1.1014920000
H     -2.7394760000   -0.8229960000    0.6487800000
    '''
    mol.basis = 'ccpvtz'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.chkfile = "rks.chk"
    if os.path.isfile(mf.chkfile):
        mf.__dict__.update(scf.chkfile.load(mf.chkfile, 'scf'))
    else:
        mf.init_guess = "chkfile"
        mf.kernel()

    # Usage 1: derive effective Fock in minimal basis, then recover MO energies and density matrix
    # minao = 'minao'
    # iomega = get_iomega()
    # F_eff, F_bare, hyb, C_ao_lo, C_ao_minao = fock_downfold(mf, omega=iomega, delta=0, lo_method='iao',
    #                                                         proj_basis=minao)

    # mo_energy, mo_coeff = solve_heff_QP(F_bare, hyb, iomega)
    # print ('mo_energy diff (eV)',(mo_energy - mf.mo_energy[:len(mo_energy)])*27.211386)

    # dip, dip_eff = test_dip_moment(mf, C_ao_minao, mo_coeff)
    # print ('dipole moment diff (D)', dip_eff-dip)

    # # Usage 2: derive effective Fock for all local orbitals on a fragment
    # minao = 'minao'
    # from pyscf.lo.iao import reference_mol
    # pmol = reference_mol(mol, minao=minao)
    # nminao = len(pmol.ao_labels())
    # idx_frag = list(range(5)) + list(range(nminao,nminao+9))
    # iomega = get_iomega()
    # F_eff, F_bare, hyb, C_ao_lo, C_ao_minao = fock_downfold(mf, omega=iomega, delta=0, local_frag=True,
    #                                                         fermi_adjust=True, idx_frag=idx_frag,
    #                                                         lo_method='lowdin', proj_basis=minao)

    # freqs = np.linspace(-1,1,101)
    # eta = 0.02
    # gf = get_gf_fromAC(F_bare, hyb, iomega, freqs, eta)

    # F_eff, F_bare, hyb, C_ao_lo, C_ao_minao = fock_downfold(mf, omega=freqs, delta=eta, local_frag=True,
    #                                                         fermi_adjust=True, idx_frag=idx_frag,
    #                                                         lo_method='lowdin', proj_basis=minao)
    # gf_real = np.zeros_like(gf)
    # nao = gf_real.shape[0]
    # for iw in range(len(freqs)):
    #     gf_real[:,:,iw] = np.linalg.inv((freqs[iw] + 1j*eta) * np.eye(nao) - F_eff[:,:,iw])

    # from fcdmft.gw.mol.gw_ac import get_g0
    # ef = get_ef(mf)
    # gf_full = get_g0(freqs, mf.mo_energy-ef, eta)
    # S = mf.get_ovlp()
    # CSC = C_ao_minao.T @ S @ mf.mo_coeff
    # print ('DOS of O atom from Heff_AC, Heff_real, full DFT')
    # for iw in range(len(freqs)):
    #     gf_tmp = CSC @ gf_full[:,:,iw] @ CSC.T
    #     print (freqs[iw], -np.trace(gf[:,:,iw].imag)/np.pi, -np.trace(gf_real[:,:,iw].imag)/np.pi,
    #            -np.trace(gf_tmp.imag)/np.pi)

    # debug only
    '''
    # Pade AC from Heff(iw) to Pade coefficients
    coeff, omega_fit = my_AC_pade_thiele_full(F_eff, iomega)

    # Get Heff(w) and scan for roots
    omega = np.linspace(-1,0.2,121)
    HeffR = solve_Heff_AC(omega, coeff, omega_fit, eta=0.01)
    '''

    # Downfold method 2:
    minao = 'minao'
    from pyscf.lo.iao import reference_mol
    pmol = reference_mol(mol, minao=minao)
    sap_type = 'dft'
    sap_basis = 'sapgrasplarge'
    idx_mo = None
    idx_mo = list(range(pmol.nao_nr()))
    F_eff, X_mat, R_mat, C_ao_lo, C_ao_minao, idx_mo \
          = fock_downfold_x(mf, lo_method='lowdin', idx_mo=idx_mo,
            proj_basis=minao, sap_type=sap_type, sap_basis=sap_basis)

    mo_energy, mo_coeff = scipy.linalg.eigh(F_eff)
    print ('mo_energy diff (eV)',(mo_energy - mf.mo_energy[idx_mo])*27.211386)

    print("exact dipole and dipole from downfolded Hamiltonian without picture change")
    dip, dip_eff = test_dip_moment(mf, C_ao_minao, mo_coeff)
    print ('dipole moment diff (D)', dip_eff-dip)

    # One can recover the exact dipole moment using exact X and R matrices
    nocc = mol.nelectron // 2
    dm_subspace = 2. * np.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    dip_back_transform = get_dip_moment_x(mol, dm_subspace, X_mat, R_mat, C_ao_lo, picture_change=True)
    print ('dipole moment diff (D)', dip_back_transform-dip)
    print("exact")
    dip_back_transform_elec = get_dip_moment_x(mol, dm_subspace, X_mat, R_mat, C_ao_lo, picture_change=True, nuc=False)

    # Recommended:
    # Using approximate X and R, e.g. from Vsap, can improve dipole moment
    if sap_type == 'dft':
        hcore_vsap = mol.intor('int1e_kin') + mf.get_vsap()
    else:
        from pyscf.scf.hf import make_sap
        from mlvk.lib.ml_helper import get_sapbasis
        hcore_vsap = mf.get_hcore() + make_sap(mf.mol, sap_basis=get_sapbasis(mol, sap_basis=sap_basis))

    # e_vsap, c_vsap = scipy.linalg.eigh(hcore_vsap, mf.get_ovlp())
    # print((c_vsap[:,idx_mo].T@ mf.get_ovlp() @ mf.mo_coeff[:,idx_mo]).diagonal())
    # print(e_vsap[idx_mo])
    # print(mf.mo_energy[idx_mo])
    # exit()
    
    hcore_vsap = C_ao_lo.T @ hcore_vsap @ C_ao_lo
    x_mat_vsap, idx_mo = get_X_matrix(hcore_vsap, n=pmol.nao_nr(), idx_mo=idx_mo)
    s_prime = np.eye(len(idx_mo)) + x_mat_vsap.T @ x_mat_vsap
    from pyscf.x2c import x2c
    r_mat_vsap = x2c._invsqrt(s_prime)
    print("X vsap")
    dip_back_transform_vsap_elec = get_dip_moment_x(mol, dm_subspace, x_mat_vsap, r_mat_vsap, C_ao_lo, picture_change=True, nuc=False)

    # hcore = mf.get_hcore()
    # hcore = C_ao_lo.T @ hcore @ C_ao_lo
    # x_mat = get_X_matrix(hcore, n=len(idx_mo), idx_mo=idx_mo)[0]
    # s_prime = np.eye(len(idx_mo)) + x_mat.T @ x_mat
    # from pyscf.x2c import x2c
    # r_mat = x2c._invsqrt(s_prime)
    # print("X hcore")
    # get_dip_moment_x(mol, dm_subspace, x_mat, r_mat, C_ao_lo, picture_change=True, nuc=False)

    # nB2 = x_mat_vsap.shape[1]
    # s_prime = np.eye(nB2) + x_mat_vsap.T @ x_mat_vsap
    # from pyscf.x2c import x2c
    # r_mat = x2c._invsqrt(s_prime)
    # mo_coeff_A = r_mat @ mo_coeff
    # mo_coeff_B = x_mat_vsap @ mo_coeff_A
    # mo_coeff_occ = C_ao_lo @ np.vstack((mo_coeff_A, mo_coeff_B))[:,:nocc]
    # dm_full = 2. * mo_coeff_occ @ mo_coeff_occ.T
    # hcore = mf.get_fock(dm=dm_full)
    # hcore = C_ao_lo.T @ hcore @ C_ao_lo
    # x_mat = get_X_matrix(hcore, n=len(idx_mo), idx_mo=idx_mo)[0]
    # s_prime = np.eye(len(idx_mo)) + x_mat.T @ x_mat
    # from pyscf.x2c import x2c
    # r_mat = x2c._invsqrt(s_prime)
    # print("X vsap + 1 iteration")
    # get_dip_moment_x(mol, dm_subspace, x_mat, r_mat, C_ao_lo, picture_change=True, nuc=False)
    

    atm_scf = get_atm_nrhf(mol)
    aoslice = mol.aoslice_by_atom()
    atm_dms = []
    atm_focks = []
    atm_hcores = []
    mo_coeff = []
    mo_occ = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in atm_scf:
            symb = mol.atom_pure_symbol(ia)

        if symb in atm_scf:
            e_hf, e, c, occ, ovlp, hcore = atm_scf[symb]
            sc = ovlp @ c
        else:  # symb's basis is not specified in the input
            raise ValueError(f"Atom {symb} not found in atomic SCF data. Please specify the basis for this atom or use a different symbol (e.g. {symb} -> {mol.atom_pure_symbol(ia)}).")

        atm_dms.append(np.dot(c*occ, c.conj().T))
        atm_focks.append(sc @ np.diag(e) @ sc.T)
        atm_hcores.append(hcore)
        mo_coeff.append(c)
        mo_occ.append(occ)

    mol_atm_fock = scipy.linalg.block_diag(*atm_focks)#- scipy.linalg.block_diag(*atm_hcores) + mf.get_hcore()
    hcore = C_ao_lo.T @ mol_atm_fock @ C_ao_lo
    x_mat = get_X_matrix(hcore, n=len(idx_mo), idx_mo=idx_mo)[0]
    s_prime = np.eye(len(idx_mo)) + x_mat.T @ x_mat
    from pyscf.x2c import x2c
    r_mat = x2c._invsqrt(s_prime)
    print("atm fock")
    get_dip_moment_x(mol, dm_subspace, x_mat, r_mat, C_ao_lo, picture_change=True, nuc=False)

    dm_atm = scipy.linalg.block_diag(*atm_dms)
    hcore = mf.get_fock(dm=dm_atm)
    hcore = C_ao_lo.T @ hcore @ C_ao_lo
    x_mat = get_X_matrix(hcore, n=len(idx_mo), idx_mo=idx_mo)[0]
    s_prime = np.eye(len(idx_mo)) + x_mat.T @ x_mat
    from pyscf.x2c import x2c
    r_mat = x2c._invsqrt(s_prime)
    print("atm density fock 1 iter")
    get_dip_moment_x(mol, dm_subspace, x_mat, r_mat, C_ao_lo, picture_change=True, nuc=False)

    dm_atm = scipy.linalg.block_diag(*atm_dms)
    mf_copy = dft.RKS(mol)
    mf_copy.xc = 'lda,vwn'
    mf_copy.grids.level = 1
    hcore = mf_copy.get_fock(dm=dm_atm)
    hcore = C_ao_lo.T @ hcore @ C_ao_lo
    x_mat = get_X_matrix(hcore, n=len(idx_mo), idx_mo=idx_mo)[0]
    s_prime = np.eye(len(idx_mo)) + x_mat.T @ x_mat
    from pyscf.x2c import x2c
    r_mat = x2c._invsqrt(s_prime)
    print("atm density LDA 1 iter")
    get_dip_moment_x(mol, dm_subspace, x_mat, r_mat, C_ao_lo, picture_change=True, nuc=False)

    # dm = mf.get_init_guess(mol, 'minao', s1e=mf.get_ovlp())
    # hcore = mf.get_fock(dm=dm)
    # hcore = C_ao_lo.T @ hcore @ C_ao_lo
    # x_mat = get_X_matrix(hcore, n=len(idx_mo), idx_mo=idx_mo)[0]
    # s_prime = np.eye(len(idx_mo)) + x_mat.T @ x_mat
    # from pyscf.x2c import x2c
    # r_mat = x2c._invsqrt(s_prime)
    # print("minao density 1 iter")
    # get_dip_moment_x(mol, dm_subspace, x_mat, r_mat, C_ao_lo, picture_change=True, nuc=False)
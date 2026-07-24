import os
import numpy as np
import time
import argparse
import json
from functools import reduce

from pyscf import gto, scf, fci, lib, cc, pbc
from pyscf.lib import logger, temporary_env

def split_xyz(xyzfile):
    """split xyz file into atom coordinates

    Args:
        xyzfile (str): xyzfile

    Returns:
        str: formatted atom coords string for pyscf mol constructor
    """    
    with open(xyzfile, 'r') as f:
        s = f.read()
    s = s.split('\n')[2:]
    return  ';'.join(s).replace('*^', 'e')

def fast_ao_fock(mo_energy, mo_coeff, ovlp):
    C_ao_mo_f = reduce(np.matmul, (ovlp, mo_coeff))
    return reduce(np.matmul, (C_ao_mo_f, np.diag(mo_energy), C_ao_mo_f.T)) # Fock matrix in AO basis from diagonal MO basis

def do_rks_calculation(mol, chkfile, **kwargs):
    """
    do DFT calculation
    
    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        **kwawrgs (dict) : calculation parameters.

    Returns:
        dict: dictionary with electronic structure data
    """   

    # Assign variables from kwargs with default values
    xc = kwargs.get('xc', 'pbe')
    mlf = {}
    
    # Hartree-Fock calculation
    mf = scf.RKS(mol)
    mf.chkfile = chkfile
    mf.xc = xc

    density_fit = kwargs.get('density_fit', False)
    if density_fit:
        mf = mf.density_fit()

    mf.kernel()

    # DFT/HF calculation outputs, cheap stuff below
    mlf['e_mf'] = mf.e_tot                         # DFT energy    
    nocc =  mol.nelectron // 2           
    mlf['nocc'] = nocc                             # occupation number/number of electrons
    mlf['mo_occ'] = np.asarray(mf.mo_occ)          # occupation number of each orbital
    mlf['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
    mlf['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
    mlf['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
    mlf['hcore'] = np.asarray(mf.get_hcore())      # hcore matrix
    mlf['dm_hf'] = np.asarray(mf.make_rdm1())      # mean field density matrix

    try:
        _, _, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    except:
        hyb = 1.0

    # Fock matrix computed with calling mf.get_fock() which calls mf.get_veff()
    mlf['fock'] = np.asarray(fast_ao_fock(mlf['mo_energy'], mlf['mo_coeff'], mlf['ovlp']))   

    # More expesnive stuff (K matrix computed 2x, J matrix computed 1x)
    # the definition of the hamiltonian is a bit tricky here, need to multiply by -0.5 to get the correct definition of vk
    vj, vk = mf.get_jk()
    mlf['vj'] = np.asarray(vj)                     # Coulomb matrix
    mlf['vk'] = -0.5*np.asarray(vk)*hyb
    mlf['vxc'] = np.asarray(mf.get_veff() - mlf['vj'] - mlf['vk'])    # exchange-correlation matrix

    mlf['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    mlf['xc'] = xc
    mlf['hyb'] = hyb

    return mf, mlf

def do_rks_kpt_calculation(cell, chkfile, **kwargs):
    """
    do DFT calculation
    
    Args:
        mol (mol): pyscf mol
        chkfile (str): pyscf/mlgf chkfile
        **kwawrgs (dict) : calculation parameters.

    Returns:
        dict: dictionary with electronic structure data
    """   

    # Assign variables from kwargs with default values
    xc = kwargs.get('xc', 'pbe')
    kmesh = kwargs['kmesh']
    scaled_center = kwargs['scaled_center']
    
    # Hartree-Fock calculation
    kpts = cell.make_kpts(kmesh, scaled_center = scaled_center)
    mlf = {'kmesh' : kmesh, 'scaled_center' : scaled_center, 'kpts' : kpts}
    mf = pbc.dft.KRKS(cell, kpts)
    mf.xc = xc
    mf.verbose = 5
    mf.chkfile = chkfile
    mf.kernel()

    # DFT/HF calculation outputs, cheap stuff below
    mlf['e_mf'] = mf.e_tot                         # DFT energy    
    nocc =  cell.nelectron // 2           
    mlf['nocc'] = nocc                             # occupation number/number of electrons
    mlf['mo_occ'] = np.asarray(mf.mo_occ)          # occupation number of each orbital
    mlf['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
    mlf['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
    mlf['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
    mlf['hcore'] = np.asarray(mf.get_hcore(cell))      # hcore matrix
    mlf['dm_hf'] = np.asarray(mf.make_rdm1())      # mean field density matrix

    try:
        _, _, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
    except:
        hyb = 1.0

    
    mlf['fock'] = mf.get_fock() # make faster later

    # More expesnive stuff (K matrix computed 2x, J matrix computed 1x)
    # the definition of the hamiltonian is a bit tricky here, need to multiply by -0.5 to get the correct definition of vk
    vj, vk = mf.get_jk()
    mlf['vj'] = np.asarray(vj)                     # Coulomb matrix
    mlf['vk'] = -0.5*np.asarray(vk)*hyb
    mlf['vxc'] = np.asarray(mf.get_veff() - mlf['vj'] - mlf['vk'])    # exchange-correlation matrix

    mlf['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    mlf['xc'] = xc
    mlf['hyb'] = hyb

    return mf, mlf



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='generate.py')    
    # system parameters are command line args
    parser.add_argument('--charge', required = False, default=0, help='system charge, assumed neutral')
    parser.add_argument('--basis', required = False, default='ccpvdz', help='basis set')
    parser.add_argument('--pseudo', required = False, default=None, help='pseudopotential for solids')
    parser.add_argument('--xyz_file', required = False, default=None, help='xyz file to start DFT calculation from (or poscar for pbc).')
    parser.add_argument('--chk_file', required = True, default=None, help='.chk file for reading and writing electronic structure data.')
    parser.add_argument('--mol_max_memory', required = False, default=4000,  type = int,  help='mol.max_memory in MB')
    parser.add_argument(
    "--kmesh",
    type=int,          
    nargs="+",           # one or more values
    help="kmesh (space-separated)",
    default = [2,2,2]
    )
    parser.add_argument(
    "--scaled_center",
    type=float,          
    nargs="+",           # one or more values
    help="for twisted k point calculation (space-separated)",
    default = [0.0,0.0,0.0]
    )
    # calculation parameters stored in json
    parser.add_argument('--json_spec', required = False, default=None, help='json file holding keyword arguments for DFT calculation')
    args = parser.parse_args()

    if args.json_spec is not None:
        assert('.json' in args.json_spec)
        with open(args.json_spec) as f:
            spec = json.load(f)
    else:
        spec = {}
    
    verbose = spec.get('verbose', 0)
    chkfile = args.chk_file
    xyzfile = args.xyz_file

    assert xyzfile is not None, 'Must supply an xyz_file for DFT calculations.'

    if '.xyz' in xyzfile:
        coords = split_xyz(xyzfile)                
        mol = gto.M(atom = coords, basis = args.basis, verbose=verbose, parse_arg=False, charge = args.charge)
        mol.max_memory = args.mol_max_memory
        mol.build()
        mf, mlf = do_rks_calculation(mol, chkfile, **spec)
        lib.chkfile.save(chkfile, 'mlf', mlf)
    elif '.vasp' in xyzfile:
        from libdmet.utils.iotools import read_poscar
        cell = read_poscar(xyzfile)     
        cell.basis = args.basis
        if args.pseudo is not None:
            cell.pseudo = args.pseudo
        cell.max_memory = args.mol_max_memory
        cell.build()
        spec['kmesh'] = args.kmesh
        spec['scaled_center'] = args.scaled_center
        mf, mlf = do_rks_kpt_calculation(cell, chkfile, **spec)
        lib.chkfile.save(chkfile, 'mlf', mlf)

import pickle
import joblib
import copyreg
from functools import cache
import pyscf.dft, pyscf.lib, pyscf.scf, pyscf.gto
from pyscf.gto import Mole
import numpy as np

from mlvk.lo.saao import get_saao
from mlvk.lib.ml_helper import get_custom_freq_gfhf_features, get_hyb_off, gGW_mo_saiao, get_sigma_fit, exclude_core_ftrs, get_g0
from mlvk.lib.ml_helper import get_core_orbital_indices, get_orbtypes_df, get_saiao_charges, get_saiao_locality
from mlvk.lib.ml_helper import get_lo_features, get_chk_saiao, load_md_saiao, load_md_saao, load_md_lao, load_md_salao, load_md_saiao_vsap #, load_md_salao2
from mlvk.lib.ml_helper import get_lok_features, load_md_salaok, get_vsap_features, get_atmf_features
from mlvk.workflow.generate import split_xyz


@cache
def get_triu_indices(n):
    # Cached to avoid repeated calls to np.triu_indices
    return np.triu_indices(n, k=1, m=None)


def array_summary(obj):
    if isinstance(obj, np.ndarray):
        return f'array(shape={obj.shape}, dtype={obj.dtype})'
    else:
        return str(obj)

def realify(mat):
    if mat.dtype == np.complex128:
        return np.column_stack([mat.real, mat.imag])
    else:
        return mat

class Data:
    # Stores data from SCF calculation on a single molecule
    # This is a wrapper around a dict.
    # You can access its fields with dot notation like Pandas.
    _internal_names_set = {'data'}
    def __setattr__(self, name, value):
        if name in self._internal_names_set:
            super().__setattr__(name, value)
        else:
            self.data[name] = value
    
    def __getattr__(self, name):
        if name not in self._internal_names_set:
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f'Attribute {name} not found')
        return super().__getattribute__(name)
    
    def __getitem__(self, key):
        if key == 'mol':
            if not self.data['mol']._built:
                self.data['mol'].build(verbose=0)
        if key == 'cell':
            if not self.data['cell']._built:
                self.data['cell'].build(verbose=0)
        return self.data[key]
    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)
    def __delitem__(self, key):
        self.data.__delitem__(key)
    def __contains__(self, key):
        return (key in self.data)
    
    def keys(self):
        return self.data.keys()
    
    
    def __init__(self, idict):
        self.data = dict(idict)
    
    @staticmethod
    def from_dict(idict):
        return Data(idict)
    
    def __repr__(self):
        return "Data(\n" + ",\n".join(str(k) + "=" + array_summary(v) for k, v in self.data.items()) + ")"

    def __str__(self):
        return self.__repr__()


    @staticmethod
    def from_kwargs(**kwargs):
        # Initialize from keyword arguments
        return Data(kwargs)

    def save(self, f):
        if hasattr(f, 'write'):
            DataPickler(f, protocol=5).dump(self.data)
        else:
            with open(f, 'wb') as fp:
                DataPickler(fp, protocol=5).dump(self.data)
    
    @staticmethod
    def load_chk(f, purge_keys = [], force_eigv_direction = True, custom_minao_path = None, basis = 'salao', charge = 0, verbose = 0, sap_type = None, blockdiag_on = 'hcore', downfold_basis='minao'):
        """primary load function for Data objects (from chk files)

        Args:
            f (str): .chk file name 
            purge_keys (list, optional): attributes to remove if wanting to save memory. Defaults to [].
            force_eigv_direction (bool, optional): force the SAIAO rotation to force the eigenvector directions. Defaults to True.
            custom_minao_path (list, optional): path for custom minao in IAO construction
            sap_type (str, optional): type of SAP to use. Defaults to None.
        Returns:
            Data: the resulting Data for the molecule, with SAIAO features in numpy format
        """      
        assert(basis in ['saiao', 'saiaovsap', 'saiao_crude','saao', 'saao_crude', 'lao_crude', 'lao', 'salao', 'salao_crude', 'salaok', 'salaok_crude', 'saiaovsap_crude', 'saiaovsap_crude_downfold']) # 
        

        if basis == 'saiao':
            mlf = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = load_md_saiao(mlf, f, force_eigv_direction = force_eigv_direction, custom_minao_path = custom_minao_path)
        if basis == 'saao':
            mlf = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = load_md_saao(mlf, f, force_eigv_direction = force_eigv_direction)

        if basis == 'lao':
            mlf = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = load_md_lao(mlf, f, force_eigv_direction = force_eigv_direction)
        if basis == 'salao':
            mlf = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = load_md_salao(mlf, f, force_eigv_direction = force_eigv_direction)
        if basis == 'salaok':
            mlf = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = load_md_salaok(mlf, f, force_eigv_direction = force_eigv_direction)

        if basis in ['salao_crude', 'saiaovsap_crude', 'saiaovsap_crude_downfold', 'saiao_crude']:
            if sap_type is None:
                sap_type = 'dft'

        if basis == 'saao_crude' and '.chk' in f:
            mlf_old = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = {}
            mol = pyscf.lib.chkfile.load_mol(f)
            mol.verbose = verbose
            mf = pyscf.dft.RKS(mol)

            if 'mo_energy' in mlf.keys():
                mlf['mo_energy_true'] = mlf_old['mo_energy'].copy()
            if 'dm_hf' in mlf.keys():
                mlf['dm_hf_true'] = mlf_old['dm_hf'].copy()
            if 'fock' in mlf.keys():
                mlf['fock_true'] = mlf_old['fock'].copy()

            true_keys = ['fock', 'vk', 'vj', 'vxc']
            for key in true_keys:
                if key in mlf_old.keys():
                    mlf[f'{key}_true'] = mlf_old[key].copy()
                    
            mlf['fock'] = mlf_old['hcore'].copy()
            mlf['ovlp'] = mlf_old['ovlp'].copy()
            mlf['hcore'] = mlf_old['hcore'].copy()
            mlf['nocc'] = mlf_old['nocc'].copy()
            mlf['mo_energy'], mlf['mo_coeff'] = mf.eig(mlf['fock'], mlf['ovlp'])
            mlf['mo_occ'] = mf.get_occ(mlf['mo_energy'], mlf['mo_coeff'])
            mlf['dm_hf'] = mf.make_rdm1(mlf['mo_coeff'], mlf['mo_occ'])
            mlf = load_md_saao(mlf, f, force_eigv_direction = force_eigv_direction, mol = mol)
            mlf['mol'] = mol

        if basis == 'lao_crude' and '.chk' in f:
            mlf_old = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = {}
            mol = pyscf.lib.chkfile.load_mol(f)
            mf = pyscf.dft.RKS(mol)

            if 'mo_energy' in mlf.keys():
                mlf['mo_energy_true'] = mlf_old['mo_energy'].copy()
            if 'dm_hf' in mlf.keys():
                mlf['dm_hf_true'] = mlf_old['dm_hf'].copy()
            if 'fock' in mlf.keys():
                mlf['fock_true'] = mlf_old['fock'].copy()

            mlf['fock'] = mlf_old['hcore'].copy()
            mlf['ovlp'] = mlf_old['ovlp'].copy()
            mlf['hcore'] = mlf_old['hcore'].copy()
            mlf['nocc'] = mlf_old['nocc'].copy()
            mlf['mo_energy'], mlf['mo_coeff'] = mf.eig(mlf['fock'], mlf['ovlp'])
            mlf['mo_occ'] = mf.get_occ(mlf['mo_energy'], mlf['mo_coeff'])
            mlf['dm_hf'] = mf.make_rdm1(mlf['mo_coeff'], mlf['mo_occ'])

            true_keys = ['fock', 'vk', 'vj', 'vxc']
            for key in true_keys:
                if key in mlf_old.keys():
                    mlf[f'{key}_true'] = mlf_old[key].copy()

            mlf = load_md_lao(mlf, f, force_eigv_direction = force_eigv_direction, mol = mol)
            mlf['mol'] = mol
        
        if basis == 'salao_crude' and '.chk' in f:
            mlf_old = pyscf.lib.chkfile.load(f, 'mlf')
            if mlf_old is None:
                mlf_old = {}
            mlf = {}
            mol = pyscf.lib.chkfile.load_mol(f)
            mol.verbose = verbose
            mol.verbose = verbose
            mf = pyscf.dft.RKS(mol)

            if 'mo_energy' in mlf_old.keys():
                mlf['mo_energy_true'] = mlf_old['mo_energy'].copy()
            if 'dm_hf' in mlf_old.keys():
                mlf['dm_hf_true'] = mlf_old['dm_hf'].copy()
            if 'fock' in mlf_old.keys():
                mlf['fock_true'] = mlf_old['fock'].copy()

            mlf['ovlp'], mlf['hcore'], mlf['nocc'] = mf.get_ovlp(), mf.get_hcore(mol), mol.nelectron // 2
            mlf['dm_hf'], mlf['vsap'], mlf['mo_energy'], mlf['mo_coeff'] = get_vsap_features(mf, mlf['ovlp'], mlf['hcore'], sap_type=sap_type)
            mlf['fock'] = mlf['vsap'] + mlf['hcore']

            true_keys = ['fock', 'vk', 'vj', 'vxc']
            for key in true_keys:
                if key in mlf_old.keys():
                    mlf[f'{key}_true'] = mlf_old[key].copy()

            mlf = load_md_salao(mlf, f, force_eigv_direction = force_eigv_direction, mol = mol, blockdiag_on = blockdiag_on)
            mlf['mol'] = mol

        if basis == 'saiaovsap_crude' and '.chk' in f:
            mlf_old = pyscf.lib.chkfile.load(f, 'mlf')
            if mlf_old is None:
                mlf_old = {}
            mlf = {}
            mol = pyscf.lib.chkfile.load_mol(f)
            mol.verbose = verbose
            mol.verbose = verbose
            mf = pyscf.dft.RKS(mol)

            if 'mo_energy' in mlf_old.keys():
                mlf['mo_energy_true'] = mlf_old['mo_energy'].copy()
            if 'dm_hf' in mlf_old.keys():
                mlf['dm_hf_true'] = mlf_old['dm_hf'].copy()
            if 'fock' in mlf_old.keys():
                mlf['fock_true'] = mlf_old['fock'].copy()

            mlf['ovlp'], mlf['hcore'], mlf['nocc'] = mf.get_ovlp(), mf.get_hcore(mol), mol.nelectron // 2
            mlf['dm_hf'], mlf['vsap'], mlf['mo_energy'], mlf['mo_coeff'] = get_vsap_features(mf, mlf['ovlp'], mlf['hcore'], sap_type=sap_type)
            mlf['fock'] = mlf['vsap'] + mlf['hcore']

            true_keys = ['fock', 'vk', 'vj', 'vxc']
            for key in true_keys:
                if key in mlf_old.keys():
                    mlf[f'{key}_true'] = mlf_old[key].copy()

            mlf = load_md_saiao_vsap(mlf, f, force_eigv_direction = force_eigv_direction, custom_minao_path = custom_minao_path, sap_type=sap_type)
            mlf['mol'] = mol

        if basis == 'saiaoatmf_crude' and '.chk' in f:
            mlf_old = pyscf.lib.chkfile.load(f, 'mlf')
            if mlf_old is None:
                mlf_old = {}
            mlf = {}
            mol = pyscf.lib.chkfile.load_mol(f)
            mol.verbose = verbose
            mol.verbose = verbose
            mf = pyscf.dft.RKS(mol)

            if 'mo_energy' in mlf_old.keys():
                mlf['mo_energy_true'] = mlf_old['mo_energy'].copy()
            if 'dm_hf' in mlf_old.keys():
                mlf['dm_hf_true'] = mlf_old['dm_hf'].copy()
            if 'fock' in mlf_old.keys():
                mlf['fock_true'] = mlf_old['fock'].copy()

            mlf['ovlp'], mlf['hcore'], mlf['nocc'] = mf.get_ovlp(), mf.get_hcore(mol), mol.nelectron // 2
            mlf['dm_hf'], mlf['vsap'], mlf['mo_energy'], mlf['mo_coeff'] = get_atmf_features(mf, mlf['ovlp'], mlf['hcore'])
            mlf['fock'] = mlf['vsap'] + mlf['hcore']

            true_keys = ['fock', 'vk', 'vj', 'vxc']
            for key in true_keys:
                if key in mlf_old.keys():
                    mlf[f'{key}_true'] = mlf_old[key].copy()

            mlf = load_md_saiao_vsap(mlf, f, force_eigv_direction = force_eigv_direction, custom_minao_path = custom_minao_path, sap_type=sap_type)
            mlf['mol'] = mol

        if basis == 'saiaovsap_crude_downfold' and '.chk' in f:
            mlf_old = pyscf.lib.chkfile.load(f, 'mlf')
            if mlf_old is None:
                mlf_old = {}
            mlf = {}
            mol = pyscf.lib.chkfile.load_mol(f)
            mol.verbose = verbose
            mol.verbose = verbose
            mf = pyscf.dft.RKS(mol)

            if 'mo_energy' in mlf_old.keys():
                mlf['mo_energy_true'] = mlf_old['mo_energy'].copy()
            if 'dm_hf' in mlf_old.keys():
                mlf['dm_hf_true'] = mlf_old['dm_hf'].copy()
            if 'fock' in mlf_old.keys():
                mlf['fock_true'] = mlf_old['fock'].copy()

            mlf['ovlp'], mlf['hcore'], mlf['nocc'] = mf.get_ovlp(), mf.get_hcore(mol), mol.nelectron // 2
            mlf['dm_hf'], mlf['vsap'], mlf['mo_energy'], mlf['mo_coeff'] = get_vsap_features(mf, mlf['ovlp'], mlf['hcore'], sap_type=sap_type)
            mlf['fock'] = mlf['vsap'] + mlf['hcore']

            true_keys = ['fock', 'vk', 'vj', 'vxc']
            for key in true_keys:
                if key in mlf_old.keys():
                    mlf[f'{key}_true'] = mlf_old[key].copy()

            mlf = load_md_saiao_vsap(mlf, f, force_eigv_direction = force_eigv_direction, custom_minao_path = custom_minao_path, reorder_saiao=True, downfold=True, sap_type=sap_type, downfold_basis=downfold_basis)
            mlf['mol'] = mol

            # add purged keys to reduce memory
            for key in mlf.keys():
                if f'{key}_downfold' in mlf.keys() and key not in purge_keys:
                    purge_keys.append(key)

            # check stability of downfolding
            basis_name = basis.split('_')[0]
            tmp = np.linalg.norm(mlf[f'S_{basis_name}_downfold'] - np.eye(mlf[f'S_{basis_name}_downfold'].shape[0]))
            if tmp > 1e-8:
                print(f'Warning! Downfolded S matrix is not identity, norm diff: {tmp} for file {f}')

        if basis == 'saiao_crude' and '.chk' in f:
            mlf_old = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = {}
            mol = pyscf.lib.chkfile.load_mol(f)
            mol.verbose = verbose
            mol.verbose = verbose
            mf = pyscf.dft.RKS(mol)

            if 'mo_energy' in mlf_old.keys():
                mlf['mo_energy_true'] = mlf_old['mo_energy'].copy()
            if 'dm_hf' in mlf_old.keys():
                mlf['dm_hf_true'] = mlf_old['dm_hf'].copy()
            if 'fock' in mlf_old.keys():
                mlf['fock_true'] = mlf_old['fock'].copy()

            mlf['ovlp'], mlf['hcore'], mlf['nocc'] = mf.get_ovlp(), mf.get_hcore(mol), mol.nelectron // 2
            mlf['dm_hf'], mlf['vsap'], mlf['mo_energy'], mlf['mo_coeff'] = get_vsap_features(mf, mlf['ovlp'], mlf['hcore'], sap_type=sap_type)
            mlf['fock'] = mlf['vsap'] + mlf['hcore']

            true_keys = ['fock', 'vk', 'vj', 'vxc']
            for key in true_keys:
                if key in mlf_old.keys():
                    mlf[f'{key}_true'] = mlf_old[key].copy()

            mlf = load_md_saiao(mlf, f, force_eigv_direction = force_eigv_direction, custom_minao_path = custom_minao_path)
            mlf['mol'] = mol

        if basis == 'salaok_crude' and '.chk' in f:
            mlf_old = pyscf.lib.chkfile.load(f, 'mlf')
            mlf = {'kpts' : mlf_old['kpts'], 'kmesh' : mlf_old['kmesh'], 'scaled_center' : mlf_old['scaled_center'], 'nkpts' : mlf_old['kpts'].shape[0]}
            cell = pyscf.pbc.lib.chkfile.load_cell(f)
            cell.verbose = verbose
            mf = pyscf.pbc.dft.KRKS(cell, mlf['kpts'])

            if 'mo_energy' in mlf_old.keys():
                mlf['mo_energy_true'] = mlf_old['mo_energy'].copy()
            if 'dm_hf' in mlf_old.keys():
                mlf['dm_hf_true'] = mlf_old['dm_hf'].copy()
            if 'fock' in mlf_old.keys():
                mlf['fock_true'] = mlf_old['fock'].copy()

            
            mlf['ovlp'] = mf.get_ovlp()
            mlf['hcore'] = mf.get_hcore(cell)
            mlf['fock'] = mlf['hcore'].copy()
            mlf['nocc'] = cell.nelectron // 2
            mlf['mo_energy'], mlf['mo_coeff'] = mf.eig(mlf['fock'], mlf['ovlp'])
            mlf['mo_occ'] = mf.get_occ(mlf['mo_energy'], mlf['mo_coeff'])
            mlf['dm_hf'] = mf.make_rdm1(mlf['mo_coeff'], mlf['mo_occ'])
            mlf['mo_energy'], mlf['mo_coeff'], mlf['mo_occ'], mlf['dm_hf'] = np.asarray(mlf['mo_energy']), np.asarray(mlf['mo_coeff']), np.asarray(mlf['mo_occ']), np.asarray(mlf['dm_hf'])

            true_keys = ['fock', 'vk', 'vj', 'vxc']
            for key in true_keys:
                if key in mlf_old.keys():
                    mlf[f'{key}_true'] = mlf_old[key].copy()

            mlf = load_md_salaok(mlf, f, force_eigv_direction = force_eigv_direction, cell = cell)
            mlf['cell'] = cell

        if custom_minao_path is not None:
            mlf['custom_minao_path'] = custom_minao_path

        md = Data({key: value for key, value in mlf.items() if key not in purge_keys})
        if 'mol' in md.data:
            if isinstance(md.data['mol'], str):
                md.data['mol'] = Mole.loads(md.data['mol'])
        if 'cell' in md.data:
            if isinstance(md.data['cell'], str):
                raise NotImplementedError
        md.fname = f
        md.basis = basis

        return md

    @staticmethod
    def load_xyz(f, purge_keys = [], force_eigv_direction = True, custom_minao_path = None, basis = 'saiao', basis_set = 'ccpvdz', charge = 0, blockdiag_on = 'hcore'):
        """primary load function for Data objects (from chk files)

        Args:
            f (str): .xyz or .vasp file name 
            force_eigv_direction (bool, optional): force the SAIAO rotation to force the eigenvector directions. Defaults to True.
            custom_minao_path (list, optional): path for custom minao in IAO construction

        Returns:
            Data: the resulting Data for the molecule, with SAIAO features in numpy format
        """      
        assert(basis in ['salao_crude', 'salao_crude', 'salaok_crude', 'saiaovsap_crude']) # , 'salaok_crude'  
        

        if basis == 'saao_crude':
            coords = split_xyz(f)                
            mol = pyscf.gto.M(atom = coords, basis = basis_set, verbose=0, parse_arg=False, charge = charge)
            mf = pyscf.dft.RKS(mol)
            mlf = {}
            mlf['ovlp'] = mf.get_ovlp()
            mlf['hcore'] = mf.get_hcore(mol)

            if 'mo_energy' in mlf.keys():
                mlf['mo_energy_true'] = mlf['mo_energy'].copy()
            if 'dm_hf' in mlf.keys():
                mlf['dm_hf_true'] = mlf['dm_hf'].copy()
            if 'fock' in mlf.keys():
                mlf['fock_true'] = mlf['fock'].copy()

            mlf['fock'] = mlf['hcore'].copy()
            mlf['mo_energy'], mlf['mo_coeff'] = mf.eig(mlf['fock'], mlf['ovlp'])
            mlf['mo_occ'] = mf.get_occ(mlf['mo_energy'], mlf['mo_coeff'])
            mlf['dm_hf'] = mf.make_rdm1(mlf['mo_coeff'], mlf['mo_occ'])
            mlf['nocc'] = mol.nelectron // 2

            mlf = load_md_saao(mlf, f, force_eigv_direction = force_eigv_direction, mol = mol)
            mlf['mol'] = mol


        if basis == 'lao_crude':
            coords = split_xyz(f)                
            mol = pyscf.gto.M(atom = coords, basis = basis_set, verbose=0, parse_arg=False, charge = charge)
            mf = pyscf.dft.RKS(mol)
            mlf = {}
            mlf['ovlp'] = mf.get_ovlp()
            mlf['hcore'] = mf.get_hcore(mol)

            if 'mo_energy' in mlf.keys():
                mlf['mo_energy_true'] = mlf['mo_energy'].copy()
            if 'dm_hf' in mlf.keys():
                mlf['dm_hf_true'] = mlf['dm_hf'].copy()
            if 'fock' in mlf.keys():
                mlf['fock_true'] = mlf['fock'].copy()

            mlf['fock'] = mlf['hcore'].copy()
            mlf['mo_energy'], mlf['mo_coeff'] = mf.eig(mlf['fock'], mlf['ovlp'])
            mlf['mo_occ'] = mf.get_occ(mlf['mo_energy'], mlf['mo_coeff'])
            mlf['dm_hf'] = mf.make_rdm1(mlf['mo_coeff'], mlf['mo_occ'])
            mlf['nocc'] = mol.nelectron // 2

            mlf = load_md_lao(mlf, f, force_eigv_direction = force_eigv_direction, mol = mol)
            mlf['mol'] = mol

        if basis == 'salao_crude':
            coords = split_xyz(f)                
            mol = pyscf.gto.M(atom = coords, basis = basis_set, verbose=0, parse_arg=False, charge = charge)
            mf = pyscf.dft.RKS(mol)
            mlf = {}
            mlf['ovlp'] = mf.get_ovlp()
            mlf['hcore'] = mf.get_hcore(mol)

            mlf['dm_hf'], mlf['vsap'], mlf['mo_energy'], mlf['mo_coeff'] = get_vsap_features(mf, mlf['ovlp'], mlf['hcore'])
            mlf['fock'] = mlf['vsap'] + mlf['hcore']
            mlf['nocc'] = mol.nelectron // 2

            mlf = load_md_salao(mlf, f, force_eigv_direction = force_eigv_direction, mol = mol, blockdiag_on = blockdiag_on)
            mlf['mol'] = mol

        if basis == 'saiaovsap_crude':
            coords = split_xyz(f)                
            mol = pyscf.gto.M(atom = coords, basis = basis_set, verbose=0, parse_arg=False, charge = charge)
            mf = pyscf.dft.RKS(mol)
            mlf = {}
            mlf['ovlp'] = mf.get_ovlp()
            mlf['hcore'] = mf.get_hcore(mol)

            mlf['dm_hf'], mlf['vsap'], mlf['mo_energy'], mlf['mo_coeff'] = get_vsap_features(mf, mlf['ovlp'], mlf['hcore'])
            mlf['fock'] = mlf['vsap'] + mlf['hcore']
            mlf['nocc'] = mol.nelectron // 2

            mlf = load_md_saiao_vsap(mlf, f, force_eigv_direction = force_eigv_direction, mol = mol, blockdiag_on = blockdiag_on)
            mlf['mol'] = mol

        if basis == 'salaok_crude':
            raise NotImplementedError

        if custom_minao_path is not None:
            mlf['custom_minao_path'] = custom_minao_path
        md = Data({key: value for key, value in mlf.items() if key not in purge_keys})
        if 'mol' in md.data:
            if isinstance(md.data['mol'], str):
                    md.data['mol'] = Mole.loads(md.data['mol'])
        md.fname = f
        md.basis = basis

        return md

    @staticmethod
    def load(f):
        # Load data from a file.
        # Takes a path-like or file-like object
        if hasattr(f, 'read'):
            return Data(pickle.load(f))
        else:
            # assume f is a path-like object
            with open(f, 'rb') as fp:
                return Data(pickle.load(fp))

    def refit_sigma(self, omega_fit):
        # Calculate dynamical features
        assert(len(self.freqs) >= len(omega_fit))
        setattr(self, 'omega_fit', omega_fit)
        sigma_fit = get_sigma_fit(self.sigmaI, self.freqs, omega_fit, freq_tol = 1.0e-8)
        for basis in ['saao', 'saiao']:
            basis_mo_rotation = f'C_mo_{basis}' 
            if basis_mo_rotation in self.data:
                sigma_fit_lmo = gGW_mo_saiao(sigma_fit, self.data[basis_mo_rotation])
                setattr(self, f'sigma_{basis}', sigma_fit_lmo)        

    def get_diag_features(self, features, ret_labels=False, exclude_core = False, k = None):
        #return np.column_stack([np.diagonal(getattr(self, f)).T for f in features])
        if exclude_core: 
            inds_core = self.inds_core
        to_stack = []
        for f in features:
            feat = self.data[f].copy()
            if k is not None:
                feat = feat[k]

            feat = np.diagonal(feat).T
            if exclude_core: 
                feat = exclude_core_ftrs(feat, inds_core, rank2 = False)
            to_stack.append(realify(feat))
        diag_features = np.column_stack(to_stack)
        
        if ret_labels:
            labels = []
            for f in features:
                feat = self.data[f]
                if feat.ndim == 3:
                    ncol = feat.shape[2]
                else:
                    # simple diagonal
                    ncol = 1
                
                if feat.dtype == np.complex128:
                    for part in ('re', 'im'): # real part, then imaginary part
                        for i in range(ncol):
                            labels.append(f'{f}_{i}_{part}' if ncol > 1 else f'{f}_{part}')
                else: # pure real
                    for i in range(ncol):
                        labels.append(f'{f}_{i}' if ncol > 1 else f)
            return diag_features, labels
        else:
            return diag_features
                        
    def calc_dyn(self, dyn_imag_freq_points, ftr_suffix = '', add_ef = True):
        basis = self.basis.split('_')[0]
        if add_ef:
            dyn_imag_freq_points = self.ef.copy() + dyn_imag_freq_points.copy()
        gf_dyn, hyb_dyn = get_custom_freq_gfhf_features(
            self.mo_energy, self[f'fock_{basis}'], self[f'C_{basis}_mo'], dyn_imag_freq_points, mlf_chkfile = getattr(self, 'fname', ''))

        setattr(self, f'gf_dyn{ftr_suffix}', gf_dyn)
        setattr(self, f'hyb_dyn{ftr_suffix}', hyb_dyn)
        
        setattr(self, f'gf_dyn{ftr_suffix}_log', np.log(np.abs(gf_dyn) + 1e-12)) # add small constant to avoid log(0)
        sign_real = np.sign(gf_dyn.real)
        sign_imag = np.sign(gf_dyn.imag)
        g0_tmp = np.log(np.abs(gf_dyn.real) + 1e-12) * sign_real + 1j * np.log(np.abs(gf_dyn.imag) + 1e-12) * sign_imag
        setattr(self, f'gf_dyn{ftr_suffix}_log_cplx', g0_tmp)

        # hyb_dyn_off = get_hyb_off(self[f'fock_{basis}'], gf_dyn, dyn_imag_freq_points)
        # setattr(self, f'hyb_dyn_off{ftr_suffix}', hyb_dyn_off)
    
    def get_offdiag_features(self, features, ret_labels = False, coulomb_screen_tol = None, coulomb_screen_basis = 'saiao', exclude_core = False, k = None):
        if "downfold" in self.basis:
            key_name = f'S_{self.basis.split("_")[0]}_downfold'
            nmo = self.data[key_name].shape[0]
        else:
            nmo = self.ovlp.shape[-1]
        
        if exclude_core:
            iu = get_triu_indices(nmo - len(self.inds_core))
        else:
            iu = get_triu_indices(nmo)
        
        def unravel_(mat):
            if mat.ndim == 3:
                return mat[iu[0], iu[1], :]
            else:
                return mat[iu]
        
        if coulomb_screen_tol is not None: 
            # print(f'Screening coulomb matrix np.abs(vj_{coulomb_screen_basis}) < {coulomb_screen_tol}')
            screen = np.abs(self.data[f'vj_{coulomb_screen_basis}']) > coulomb_screen_tol
           
        else:
            screen = np.ones((nmo, nmo), dtype = bool)

        def get_ftr(f):
            if k is not None:
                return self.data[f][k]
            else:
                return self.data[f]
        
        if exclude_core:
            inds_core = self.inds_core
            # print(screen.shape, inds_core)
            screen = unravel_(exclude_core_ftrs(screen, inds_core))
            offdiag_features = np.column_stack([realify(unravel_(exclude_core_ftrs(get_ftr(f), inds_core))[screen]) for f in features])
        else:
            screen = unravel_(screen)
            offdiag_features = np.column_stack([realify(unravel_(get_ftr(f))[screen]) for f in features])
        
        if ret_labels:
            labels = []
            for f in features:
                feat = self.data[f]
                if feat.ndim == 3:
                    ncol = feat.shape[2]
                else:
                    ncol = 1
                
                if feat.dtype == np.complex128:
                    for part in ('re', 'im'): # real part, then imaginary part
                        for i in range(ncol):
                            labels.append(f'{f}_{i}_{part}' if ncol > 1 else f'{f}_{part}')
                else: # pure real
                    for i in range(ncol):
                        labels.append(f'{f}_{i}' if ncol > 1 else f)
            return offdiag_features, labels
        else:
            return offdiag_features



import os
import numpy as np
import pandas as pd
from ._data import Data
import pathlib
import re
import warnings


def get_conf_nums(fnames):
    stems = [pathlib.Path(f).stem for f in fnames]
    
    try:
    # This regex gets the last contiguous sequence of digits in a string.
    # The string must end in a digit.
        conf_nums = [int(re.match(r'.*?(\d+)\Z', stem).group(1)) for stem in stems]
    except AttributeError:
        # warnings.warn(f'Could not parse numbers from file stems {stems}')
        return [i for i in range(len(stems))]
    return conf_nums
    

class Dataset:
    """Holds a list of data from DFT and/or GW calculations as Data for each DFT or DFT+GW calculation
    Attributes:
        fnames (list): List of names of data files
        loaded (dict): Dictionary containing loaded data
        data_format (str): Format of data files
        
    """
    _format_extension_tbl = {'joblib': '.joblib', 'internal': '.mdt', 'chk' : '.chk'}
    
    
    def __init__(self, iterable, loaded = {}, data_format = 'chk', load_data = True, purge_keys = [], custom_minao_path = None, basis = 'saiao'):
        self.fnames = list(iterable)
       

        self.purge_keys = purge_keys
        self.basis = basis
        self.byfname = True
        
        self.loaded = loaded
        self.data_format = data_format
        self.load_data = load_data
        if self.data_format not in Dataset._format_extension_tbl.keys():
            raise ValueError(f'Unknown data format {self.data_format}; acceptable values are {Dataset._format_extension_tbl.keys()}')

        self.custom_minao_path = custom_minao_path

    
    def get_by_fname(self, fname):
        if fname in self.loaded:
            return self.loaded[fname]
        else:
            if self.data_format == 'joblib':
                dat = Data.load_joblib(fname)
            elif self.data_format == 'chk':
                dat = Data.load_chk(fname, purge_keys = self.purge_keys, custom_minao_path = self.custom_minao_path, basis = self.basis)
            elif self.data_format == 'internal':
                dat = Data.load(fname)
            else:
                raise ValueError(f'Unknown data format {self.data_format}; acceptable values are {Dataset._format_extension_tbl.keys()}')
            if self.load_data:
                self.loaded[fname] = dat
            return dat
        
    def __getitem__(self, idx):
        if self.byfname:
            fname = self.fnames[idx]
            return self.get_by_fname(fname)
        elif self.bylist:
            return self.loaded[str(idx)]

    def __setitem__(self, idx, value):
        self.loaded[self.fnames[idx]] = value
    
    def __len__(self):
        return len(self.fnames)
    
    def __iter__(self):
        return (self.get_by_fname(fname) for fname in self.fnames)
    
    @staticmethod
    def from_files(file_list, dropout_files = [], data_format = 'chk', load_data = True, purge_keys = [], custom_minao_path = None, basis = 'saiao'):
        dset = Dataset([filename for filename in file_list if filename not in dropout_files], data_format=data_format, load_data = load_data, purge_keys = purge_keys, custom_minao_path = custom_minao_path, basis = basis)
        dset.byfname = True
        dset.bylist = False
        return dset

    @staticmethod
    def from_data(data_list):
        dset = Dataset([str(i) for i in range(len(data_list))], basis = 'saao_scf')
        dset.loaded = {str(i) : Data.from_dict(data) for i, data in enumerate(data_list)}
        dset.byfname = False
        dset.bylist = True
        return dset

    def num_orbs(self):
        return self[0]['P'].shape[0]
    
    def get_diag_features(self, features, as_dataframe=False, exclude_core = False):
        assert(len(features) > 0)
        assert(len(self.fnames) > 0)
        if as_dataframe:
            _, feature_labels = self[0].get_diag_features(features, ret_labels=True)
            features = np.row_stack([self[i].get_diag_features(features, exclude_core = exclude_core) for i in range(len(self.fnames))])
            return pd.DataFrame(features, columns=feature_labels)
        else:
            return np.row_stack([self[i].get_diag_features(features, exclude_core = exclude_core) for i in range(len(self.fnames))])
    
    def get_offdiag_features(self, features, as_dataframe = False, coulomb_screen_tol = None, coulomb_screen_basis = 'saiao', exclude_core = False):
        assert(len(features) > 0)
        assert(len(self.fnames) > 0)

        if as_dataframe:
            _, feature_labels = self[0].get_offdiag_features(features, ret_labels=True)
            features = np.row_stack([self[i].get_offdiag_features(features, coulomb_screen_tol = coulomb_screen_tol, coulomb_screen_basis = coulomb_screen_basis, exclude_core = exclude_core) for i in range(len(self.fnames))])
            return pd.DataFrame(features, columns=feature_labels)
        else:
            return np.row_stack([self[i].get_offdiag_features(features, coulomb_screen_tol = coulomb_screen_tol, coulomb_screen_basis = coulomb_screen_basis, exclude_core = exclude_core) for i in range(len(self.fnames))])


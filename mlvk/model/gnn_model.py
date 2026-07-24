import os
import numpy as np
import joblib
import warnings
import shutil

import torch
import torch_geometric
import mlvk
from mlvk.data import Dataset, Data
from mlvk.model.pytorch.data import GraphDataset, Graph, unravel_rank2, construct_vk_from_graph, reconstruct_rank2, unravel_rank3
from mlvk.model.pytorch.helpers import predict_wrapper_graph_ensemble, get_graph_ensemble_mean, get_graph_ensemble_uncertainty
from mlvk.lib.ml_helper import get_features_delta, extract_ao_features, get_features_delta_samebasis

from mlvk.model.preprocessing import DataScaler, FrobeniusScaler, FrobeniusSineScaler, batched_update_standardscaler
from joblib import Parallel, delayed

import pyscf.dft
from pyscf import lib

try:
    from mpi4py import MPI
    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    print('mpi4py import failed')
    comm = None
    size = 1
    rank = 0

try:
    torch.serialization.add_safe_globals([
    mlvk.model.pytorch.data.Graph, 
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
    ])
    if rank == 0:
        print('allowed mlvk.data objects to be pickled!')
except:
    if rank == 0:
        print('mlvk.data objects not allowed to be pickled!')


"""
GNN_Model is the central object for training Vk GNN and predicting with it
a wrapper for predicting the self energy with GNN ensembles
parameters are stored in model_ii_states or model_ij_states as lists of OrderedDictionaries
for GNN ensemble of n networks, the self-energy can be predicted in predict_vk() method by taking the mean of n predictions
for GNN ensemble of n networks, the self-energy uncertainty can be estimated in uncertainty_vk() method by taking a sample standard error of the mean with n-1 degrees of freedom
"""

def realify(mat):
    if mat.dtype == np.complex128:
        return np.column_stack([mat.real, mat.imag])
    else:
        return mat


def process_file(chk_file, custom_minao_path, basis, dynamical_ftr_freqs, blockdiag_on, downfold_basis='minao', sap_type=None):
    print(chk_file, blockdiag_on)
    mlf = Data.load_chk(chk_file, custom_minao_path=custom_minao_path, basis=basis, downfold_basis=downfold_basis, blockdiag_on = blockdiag_on, sap_type=sap_type)
    if dynamical_ftr_freqs is not None:
        mlf.calc_dyn(dynamical_ftr_freqs)
    return mlf

def boost_process_file(chk_file, predict_vk_chain):
    print(chk_file, flush = True)
    # try:
    mlf = predict_vk_chain(chk_file, file_ref = chk_file)
    return mlf
    # except Exception as e:
    #     print(f'ERROR on file: {chk_file}: ', str(e))
    #     return None


def process_molecule(k, mol, data_to_graph, feature_list_ii, feature_list_ij, transformer, targets, torch_data_root):
    gx = data_to_graph(mol, feature_list_ii, feature_list_ij, transformer, extras=True)
    gy = data_to_graph(mol, targets, targets)
    gx.vk_ii = gy.x
    gx.vk_ij = gy.edge_attr
    if hasattr(gx, 'dm'):
        vk = construct_vk_from_graph(gx.vk_ii, gx.vk_ij, gx)
        gx.e2 = torch.einsum('ijn,ji->n', vk, reconstruct_rank2(gx.dm, gx.nmo))

    torch.save(gx, f'{torch_data_root}/raw/data_{k}.pt')
    return

def get_features_for_scaler(mlf, feature_list_ii, feature_list_ij, mask_screen):
    xii = []
    xij = []
    mask_no_diag = mask_screen.copy()
    mask_no_diag[np.arange(mask_screen.shape[0]), np.arange(mask_screen.shape[0])] = 0
    for f in feature_list_ii:
        xii.append(realify(np.diagonal(mlf[f].T)[np.diagonal(mask_screen)]))
    for f in feature_list_ij:
        xij.append(realify(mlf[f][mask_no_diag]))


    return np.column_stack(xii), np.column_stack(xij)



class GNN_Model:

    def __init__(self, feature_list_ii, feature_list_ij, targets, torch_data_root,
        transformer = None, exclude_core = False, basis = 'salao',
        ensemble_n = 1, model_alias = 'GNN', model_kwargs = {}, loss_kwargs = {}, 
        edge_cutoff = 0.0, edge_cutoff_features = None, basis_set = 'sto3g', custom_minao_path = None, precision = 'double',
        sap_type = None, downfold_basis_set = 'minao',
        dynamical_ftr_freqs = None, blockdiag_on = 'hcore'):
        
        self.loss_kwargs = loss_kwargs
        self.torch_data_root = torch_data_root
        
        self.edge_cutoff = edge_cutoff
        self.edge_cutoff_features = edge_cutoff_features

        self.feature_list_ii = feature_list_ii
        self.feature_list_ij = feature_list_ij
        self.feat_list_iijj = list(set(['_'.join(f.split('_')[:2]) for f in feature_list_ij if 'iijj' in f]))
        self.model_alias = model_alias

        self.model_kwargs = model_kwargs
        self.custom_minao_path = custom_minao_path

        self.dynamical_ftr_freqs = dynamical_ftr_freqs
        self.blockdiag_on = blockdiag_on
        self.sap_type = sap_type

        self.precision = precision
        self.downfold_basis_set = downfold_basis_set
        # if  model_precision != loss_precision:
        #     warnings.warn(f'Precision of model ({model_precision}) and precision of loss function ({loss_precision}) differ, modifying loss precision to be the model precision')
        #     self.loss_kwargs['precision'] = model_precision

        if type(targets) != list:
            self.targets = [targets]
        else:
            self.targets = targets

        self.exclude_core = exclude_core # only relevant for training purposes, not prediction
        self.basis = basis
        self.transformer = transformer

        self.ensemble_n = ensemble_n
        self.is_directed = False
        self.basis_set = basis_set
    
    def convert_precision(self, tensor):
        if self.precision == 'double':
            return tensor.double()
        if self.precision == 'float':
            return tensor.float()
        if self.precision == 'half':
            return tensor.half()
        
        return tensor.double()
    
    def data_to_graph(self, moldatum, features_ii, features_ij, transformer = None, extras = False):
        """Important function for creating the DFT graphs from moldatum objects

        Args:
            moldatum (Data)
            features_ii (list of string): feautres on nodes
            features_ij (list of string): features on edges
            transformer (custom scaler object): transformer for nodes and edges

        Returns:
            Graph: the PyG graph object
        """
        this_basis = self.basis.split('_')[0]
        if "downfold" in self.basis:
            key_name = f'hcore_{this_basis}_downfold'
            norbs = moldatum[key_name].shape[0]
        else: 
            norbs = moldatum['hcore'].shape[-1]
        iu = np.array(np.triu_indices(norbs, 1)) # for the assignment of full vk
        x = moldatum.get_diag_features(features_ii)
        edges = moldatum.get_offdiag_features(features_ij)        
        
        toremove_edges, toremove = np.empty(edges.shape[0], dtype = 'bool'), np.empty(x.shape[0], dtype = 'bool')
        toremove_edges[:], toremove[:] = False, False

        if self.exclude_core:
            toremove = np.array([i in moldatum['inds_core'] for i in range(x.shape[0])])
            x = x[~toremove,:]
            toremove_edges = toremove_edges | toremove[iu[0,:]] | toremove[iu[1,:]]
        
        # for the graph where core nodes may be removed
        edge_index = np.array(np.triu_indices(x.shape[0], 1))
        to_remove_graph_edge = np.empty(edge_index.shape[1], dtype = 'bool')
        to_remove_graph_edge[:] = False
        
        if self.edge_cutoff > 0.0 and self.edge_cutoff_features is not None:
            toremove_edges_new = np.max(np.abs(moldatum.get_offdiag_features(self.edge_cutoff_features)), axis = 1) < self.edge_cutoff
            toremove_edges = toremove_edges | toremove_edges_new

            to_remove_graph_edge_new = np.max(np.abs(moldatum.get_offdiag_features(self.edge_cutoff_features, exclude_core = self.exclude_core)), axis = 1) < self.edge_cutoff
            to_remove_graph_edge = to_remove_graph_edge | to_remove_graph_edge_new

        iu = iu[:,~toremove_edges]    
        edges = edges[~toremove_edges,:]
        edge_index = edge_index[:,~to_remove_graph_edge]
        
        if transformer is not None:
            if getattr(transformer, 'require_norm', False):
                norms_ii = np.linalg.norm(np.stack([moldatum[f] for f in features_ii]), axis = (1,2))
                norms_ij = np.linalg.norm(np.stack([moldatum[f] for f in features_ij]), axis = (1,2))
            else:
                norms_ii, norms_ij = None, None
            x, edges = transformer.transform(x, edges, norms_ii = norms_ii, norms_ij = norms_ij)

        # make graph undirected, don't need this for multimodal where there is no explicit graph messagepassing
        if not self.is_directed:
            edges = np.vstack((edges, edges))
            edge_index = np.hstack((edge_index, np.array([edge_index[1,:], edge_index[0,:]]))) 
            iu = np.hstack((iu, np.array([iu[1,:], iu[0,:]]))) 

        x_tensor = self.convert_precision(torch.from_numpy(x))
        edge_attr_tensor = self.convert_precision(torch.from_numpy(edges))

        graph = Graph(x = x_tensor, edge_index = torch.from_numpy(edge_index), edge_attr = edge_attr_tensor)
        graph.nodes_plus_edges = x_tensor.size(0) + edge_attr_tensor.size(0)

        
        graph.node_indices_nonzero = torch.from_numpy(np.arange(norbs)[~toremove])
        graph.edge_indices_nonzero = torch.from_numpy(iu).T
        graph.undirected = not self.is_directed
        graph.fname = str(getattr(moldatum, 'fname', 'null'))
        graph.nmo = norbs
        graph.nout = len(self.targets)
        graph.nocc = moldatum.nocc
        graph.inds_core = torch.tensor(moldatum['inds_core'])
        graph.inds_val = torch.tensor([i for i in range(graph.nmo) if i not in moldatum['inds_core']])
        graph.mask_core = torch.isin(graph.edge_index[0], graph.inds_core) & torch.isin(graph.edge_index[1], graph.inds_core)
        graph.mask_val = torch.isin(graph.edge_index[0], graph.inds_val) & torch.isin(graph.edge_index[1], graph.inds_val)
        graph.mask_cv = (torch.isin(graph.edge_index[0], graph.inds_core) & torch.isin(graph.edge_index[1], graph.inds_val)) | \
                        (torch.isin(graph.edge_index[0], graph.inds_val) & torch.isin(graph.edge_index[1], graph.inds_core))
        output_scaffold = torch.zeros((norbs, norbs, len(self.targets)), dtype = graph.x.dtype, device = graph.x.device)

        def get_key_name(target):
            if 'downfold' in self.basis:
                return f'{target}_{this_basis}_downfold'
            else:
                return f'{target}_{this_basis}'

        # TODO: handle later
        for i, f in enumerate(self.targets):
            if 'fock_true' in f:
                key_name = get_key_name('hcore')
                output_scaffold[:,:,i] = self.convert_precision(torch.from_numpy(moldatum[key_name]))

        graph.output_scaffold = unravel_rank3(output_scaffold)

        if extras:
            for target in self.targets:
                if 'delta_fock' in target:
                    key_name = get_key_name('fock')
                    graph.prev_fock = self.convert_precision(unravel_rank2(torch.tensor(moldatum[key_name])))
            

            if f'dm_hf_true_{this_basis}' in moldatum.keys():
                if 'downfold' in self.basis:
                    key_name = f'dm_hf_true_{this_basis}_downfold_exact'
                else:
                    key_name = f'dm_hf_true_{this_basis}'
                graph.dm = self.convert_precision(unravel_rank2(torch.from_numpy(moldatum[key_name])))
            else:
                key_name = get_key_name('dm')
                graph.dm = self.convert_precision(unravel_rank2(torch.from_numpy(moldatum[key_name])))

            dipole = []
            for xyz in range(3):
                xx = ['x', 'y', 'z'][xyz]
                if f'dipole_{xx}_{this_basis}' in moldatum.keys():
                    if 'downfold' in self.basis:
                        key_name = f'dipole_{xx}_{this_basis}_downfold_exact'
                    else:
                        key_name = f'dipole_{xx}_{this_basis}'
                    dipole.append(self.convert_precision(unravel_rank2(torch.from_numpy(moldatum[key_name]))))
            graph.dipole_x, graph.dipole_y, graph.dipole_z = dipole[0], dipole[1], dipole[2]
                            
            if f'fock_true_{this_basis}' in moldatum.keys():
                if this_basis not in ['saiao', 'mo', 'lao', 'salao', 'salao2','saiaovsap']:
                    key_name = get_key_name('S')
                    graph.eigh_X = self.convert_precision(torch.inverse(torch.linalg.cholesky(torch.from_numpy(moldatum[key_name]))))
                    key_name = get_key_name('fock_true')
                    F_standard = graph.eigh_X @ torch.from_numpy(moldatum[key_name]) @ graph.eigh_X.T
                    graph.mo_energy = self.convert_precision(torch.linalg.eigvalsh(F_standard))
                    graph.eigh_X = self.convert_precision(unravel_rank2(graph.eigh_X))

                else:
                    if 'downfold' in self.basis:
                        key_name = f'fock_true_{this_basis}_downfold_exact'
                    else:
                        key_name = f'fock_true_{this_basis}'
                    F_standard =  torch.from_numpy(moldatum[key_name])
                    mo_energy, mo_coeff = torch.linalg.eigh(F_standard)
                    graph.mo_energy = self.convert_precision(mo_energy)
                    graph.mo_coeff = self.convert_precision(unravel_rank2(mo_coeff))



            # else:
            #     graph.prev_fock = None
            # if this_basis not in ['saiao', 'mo', 'lao']:
            #     graph.ovlp = self.convert_precision(unravel_rank2(torch.tensor(moldatum[f'S_{this_basis}'])))



            graph.atomic_idx = torch.tensor(moldatum['atomic_index'])
            graph.n_atoms = moldatum['n_atoms']
            graph.extras = True

        else:
            graph.extras = False

        return graph

    def load_dset(self, train_files, n_jobs = 1):
        """batched fitting of StandardScaling of ii and ij features for large datasets

        Args:
            files: list of chk files for training
        Returns:
            self
        """       
       
        self.train_files = train_files
        mlf_list = []
        # Define a function to process each file

        # Parallelize the loop with Joblib
        mlf_list = Parallel(n_jobs=n_jobs, backend = 'multiprocessing')(delayed(process_file)(chk_file, self.custom_minao_path, self.basis, self.blockdiag_on, self.sap_type) for chk_file in train_files)                
        dset = Dataset.from_data(mlf_list)
        
        if type(self.transformer) == DataScaler:
            xii = dset.get_diag_features(self.feature_list_ii, exclude_core = self.exclude_core)
            xij = dset.get_offdiag_features(self.feature_list_ij, exclude_core = self.exclude_core)

            if self.edge_cutoff > 0.0 and self.edge_cutoff_features is not None:
                screen_features = dset.get_offdiag_features(self.edge_cutoff_features, exclude_core = self.exclude_core)
                toremove = np.max(np.abs(screen_features), axis = 1) < self.edge_cutoff
                xij = xij[~toremove,:]
        else:
            xii, xij = None, None

        self.transformer.fit(xii, xij)

        if os.path.exists(self.torch_data_root):
            shutil.rmtree(self.torch_data_root)
            
        os.makedirs(f'{self.torch_data_root}/raw')


        # Parallelize the loop with Joblib
        _ = Parallel(n_jobs=n_jobs, backend = 'multiprocessing')(delayed(process_molecule)(k, mol, self.data_to_graph, self.feature_list_ii, self.feature_list_ij, self.transformer, self.targets, self.torch_data_root) for k, mol in enumerate(dset))
        
        self.data = GraphDataset(f'{self.torch_data_root}', in_memory = False)
        return self



    def load_dset_mpi(self, train_files):
        """batched fitting of StandardScaling of ii and ij features for large datasets

        Args:
            train_files: list of chk files for training
        Returns:
            self
        """

        self.train_files = train_files
        
        mlf_list = []

        # Split files among MPI ranks
        data_chunk_idxs = np.array_split(np.arange(len(self.train_files)), size)
        if rank == 0:
            print('N training data: ', len(self.train_files), ' split into: ', data_chunk_idxs)

        mlf_list_local = [process_file(self.train_files[idx], self.custom_minao_path, self.basis, self.dynamical_ftr_freqs, self.blockdiag_on, self.downfold_basis_set, self.sap_type) for idx in data_chunk_idxs[rank]]
        print(f'rank {rank} finished making LO features!', flush = True)
        comm.barrier()

        if type(self.transformer) == DataScaler:
            for i, mlf in enumerate(mlf_list_local):
                if self.edge_cutoff > 0.0 and self.edge_cutoff_features is not None:
                    mask_screen = ~(np.max(np.stack([np.abs(mlf[f]) for f in self.edge_cutoff_features], axis = -1), axis = -1) < self.edge_cutoff)
                    mask_screen[np.arange(mask_screen.shape[0]), np.arange(mask_screen.shape[0])] = 1
                else:
                    if "downfold" in self.basis:
                        key_lo = f'hcore_{self.basis.split("_")[0]}_downfold'
                        mask_screen = np.ones(mlf[key_lo].shape, dtype = bool)
                    else:
                        mask_screen = np.ones(mlf['hcore'].shape, dtype = bool)

                if self.exclude_core:
                    mask_core = np.ones_like(mask_screen)
                    mask_core[mlf.inds_core,:] = mask_core[:,mlf.inds_core] = 0
                    mask_screen = mask_core*mask_screen

                # so that xii and xij are scaled the same if they are both ii and ij features
                xii, xij = get_features_for_scaler(mlf, self.feature_list_ii, self.feature_list_ij, mask_screen)
                # xii, xij = np.stack([mlf[f][mask_screen] for f in self.feature_list_ii], axis = -1), np.stack([mlf[f][mask_screen] for f in self.feature_list_ij], axis = -1)

                if i == 0:
                    self.transformer = self.transformer.fit(xii, xij)
                else:
                    self.transformer = self.transformer.batch_update(xii, xij)
        
            transformers_gather = comm.gather(self.transformer)
            if rank == 0:
                
                for i in range(len(transformers_gather)):
                    print(f'DataScaler transformers gathered for rank {i}', flush = True)
                    if i == 0:
                        new_transformer = transformers_gather[i]
                    else:
                        new_transformer.transformer_xii = batched_update_standardscaler(new_transformer.transformer_xii, transformers_gather[i].transformer_xii)
                        new_transformer.transformer_xij = batched_update_standardscaler(new_transformer.transformer_xij, transformers_gather[i].transformer_xij)
                
                transformers_gather = new_transformer # for broadcast from rank 0

            comm.Barrier()
            transformers_gather = comm.bcast(transformers_gather, root=0)
            comm.Barrier()
            self.transformer = transformers_gather
            for iftr, ftr in enumerate(self.feature_list_ii):
                if ftr in self.feature_list_ij:
                    jftr = self.feature_list_ij.index(ftr)
                    if rank == 0:
                        print(f'Unifying {ftr} stats for ii,ij: {iftr},{jftr}')
                    self.transformer.unify_shared_feature(iftr, jftr)
            
        
        elif self.transformer is None:
            pass
        else:
            self.transformer.fit(None, None)

        if rank == 0:
            if os.path.exists(self.torch_data_root):
                shutil.rmtree(self.torch_data_root)

            os.makedirs(f'{self.torch_data_root}/raw')

        comm.barrier()
        

        for i, k in enumerate(data_chunk_idxs[rank]):
            process_molecule(k, mlf_list_local[i], self.data_to_graph, self.feature_list_ii, self.feature_list_ij, self.transformer, self.targets, self.torch_data_root)

        print(f'rank {rank} finished converting data to torch!', flush = True)
        comm.barrier()
        self.data = GraphDataset(f'{self.torch_data_root}', in_memory=False)

        return self

    def mount_predict_data(self, file):  
        """prepare DFT features for prediction

        Args:
            mlf_chkfile (str): chkfile with DFT calculation
        """        
        if '.xyz' in file or '.vasp' in file:
            assert 'crude' in self.basis
            self.pdata = Data.load_xyz(file, basis = self.basis, basis_set = self.basis_set, blockdiag_on = getattr(self, 'blockdiag_on', 'hcore'))
        elif '.chk' in file:
            self.pdata = Data.load_chk(file, custom_minao_path=getattr(self, 'custom_minao_path', None), basis=self.basis, blockdiag_on = getattr(self, 'blockdiag_on', 'hcore'))
        else:
            raise
    
    def predict_vk(self, mlf_chkfile, remount_mlf_chkfile = True, exclude_core = False): # 
        """vk prediction

        Args:
            mlf_chkfile (string): chkfile with DFT calculation to compute self-energy from
            remount_mlf_chkfile (bool, optional): remount chkfile into the pgraph stored in the object. Defaults to True.
        Returns:
            np.float64: predicted vj, vk, vxc, etc tensor in SAIAO basis (nmo x nmo x ntargets)
        """    
        if remount_mlf_chkfile:
            self.mount_predict_data(mlf_chkfile)
        if getattr(self, 'dynamical_ftr_freqs', None) is not None:
            self.pdata.calc_dyn(self.dynamical_ftr_freqs)
        self.pgraph = self.data_to_graph(self.pdata, self.feature_list_ii, self.feature_list_ij, self.transformer)
        # torch.save(self.pgraph, 'test.pt')
        # self.pgraph = torch.load('test.pt')
        vk_matrices =  predict_wrapper_graph_ensemble(self.pgraph, self.model_states, self.model_alias, **self.model_kwargs)
        vk = get_graph_ensemble_mean(vk_matrices) # in saiao
        return vk

    def uncertainty_vk(self, mlf_chkfile, remount_mlf_chkfile = True):
        """self-energy uncertainty

        Args:
            mlf_chkfile (string): chkfile with DFT calculation to compute self-energy from
            remount_mlf_chkfile (bool, optional): remount chkfile into the pgraph stored in the object. Defaults to True.
        Returns:
            np.complex64: estimated self-energy uncertainty tensor in SAIAO basis (nmo x nmo x nomega)
        """    
        if self.ensemble_n == 1:
            return None
        
        if remount_mlf_chkfile:
            self.mount_predict_dset(mlf_chkfile)

        self.pgraph = self.data_to_graph(self.pdset[0], self.feature_list_ii, self.feature_list_ij, self.transformer_xii, self.transformer_xij)
        
        vk_matrices = predict_wrapper_graph_ensemble(self.pgraph, self.model_states, self.model_alias, **self.model_kwargs)
        return get_graph_ensemble_uncertainty(vk_matrices)
    
    def dump(self, filename):
        """pickile the object

        Args:
            filename (string): joblib file to pickle to
        """        
        
        if hasattr(self, 'model'):
            self.model_state = self.model.state_dict()
            delattr(self, 'model')

        if hasattr(self, 'models'):
            self.model_states = [m.state_dict() for m in self.models]
            delattr(self, 'models')

        joblib.dump(self, filename)

from typing import Dict, List

import os

import torch
from torch_geometric.data import Dataset as torchDataset
from torch_geometric.data import Data as torchData
# torch.set_default_dtype(torch.float64)

class GraphDataset(torchDataset):
    """A collection of DFT graph objects

    Args:
        torchDataset (Dataset from torch_geometric): inherits most functionality from torch_geometric
    """    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, in_memory = False, indices_ = None, rank = None):
        
        self.rank = rank
        super().__init__(root, transform, pre_transform, pre_filter)
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.in_memory = in_memory
        
        if self.in_memory:
            self.data = self.load_in_memory()
        
        else:
            self.data = None
        
        # print('pytorch dset raw_dir: ', self.raw_dir)
        if indices_ is None:
            length = len(os.listdir(f'{self.raw_dir}'))
            self.indices_ = list(range(length))
        else:
            self.indices_ = indices_
        if not rank is None:
            self.processed_dir = f'{root}/processed_{rank}'
        
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return []

    @property
    def processed_dir(self):
        if not self.rank is None:
            self.processed_dir = f'{self.root}/processed_{self.rank}'
        else:
            self.processed_dir = f'{self.root}/processed'

    @processed_dir.setter
    def processed_dir(self, rank):
        return f'{self.root}/processed_{rank}'
    
    def len(self):
        return len(self.indices_)

    def download(self):
        pass

    def load_in_memory(self):
        return [torch.load(os.path.join(self.raw_dir, f'data_{idx}.pt'))  for idx in range(self.len())]
    
    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.in_memory:
            return self.data[idx]
        else:
            return torch.load(os.path.join(self.raw_dir, 
                                 f'data_{self.indices_[idx]}.pt'))   

class Graph(torchData):
    """A DFT graph object

    Args:
        torchData (Data from torch_geometric): inherits most functionality from torch_geometric
    """    
    def __init__(self, x = None, edge_attr=None, edge_index=None):
        super().__init__(x = x, edge_attr=edge_attr, edge_index=edge_index)
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """

def construct_vk_from_graph(node_values, edge_values, graph, device = None):
    """Reconstruct the full rank3 self-energy tensor from node self-energy and edge_self-energy

    Args:
        tensor1 (torch.float64): the node self-energy
        tensor2 (torch.float64): the edge self-energy
        graph (gnn_orchestrator.GraphData object): the graph object that stores the indices of nodes and edges that have not been removed from the DFT graph

    Returns:
        torch.float64: vk
    """    
    edge_indices, node_indices = graph.edge_indices_nonzero, graph.node_indices_nonzero
    # Initialize the rank-3 tensor with zeros

    result_tensor = torch.zeros(graph.nmo, graph.nmo, graph.nout, dtype=node_values.dtype, device=node_values.device)

    # Fill the diagonal with tensor1 entries
    result_tensor[node_indices, node_indices,:] = node_values

    # Fill the upper and lower triangle with tensor2 entries (graph is undirected so we don't have to flip stuff, though it could save computation)
    result_tensor[edge_indices[:,0], edge_indices[:,1],:] = edge_values

    # saves memory if tensor1 and tensor2 are not loaded onto GPU before this function call
    if not device is None:
        result_tensor = result_tensor.to(device)

    return result_tensor


def construct_vk_from_graph_kpts(tensor1, tensor2, node_indices_nonzero, edge_indices_nonzero, nmo, device = None):
    """Reconstruct the full rank3 self-energy tensor from node self-energy and edge_self-energy

    Args:
        tensor1 (torch.float64): the node self-energy
        tensor2 (torch.float64): the edge self-energy
        node_indices_nonzero (torch.int): indices in full nmo where graph nodes get assigned 
        edge_indices_nonzero (torch.int): indices in full tensor where edge_indices_nonzero get assigned 
        nmo (int): nmo

    Returns:
        torch.float64: cat[sigma(iw).real, sigma(iw).imag]
    """    
    # from mlvk.data import Data
    # from mlvk.lib.utils import plot_performance
    # fock_true = Data.load_chk('si.chk', basis = 'salaok_crude')['fock_true_salaok']
    


    nij = edge_indices_nonzero.shape[0] // 2
    # Initialize the rank-3 tensor with zeros
    result_tensor = torch.zeros(nmo, nmo, 2, dtype=tensor1.dtype, device=tensor1.device)

    # Fill the diagonal with tensor1 entries
    result_tensor[node_indices_nonzero, node_indices_nonzero, [0]] = tensor1[:,0] # diagonal must be real

    # Fill the upper and lower triangle with tensor2 entries (graph is undirected so we don't have to flip stuff, though it could save computation)
    result_tensor[edge_indices_nonzero[:,0], edge_indices_nonzero[:,1], 0] = tensor2[:, 0] # real parts
    
    # imag part, fock is hermitian
    result_tensor[edge_indices_nonzero[:nij,0], edge_indices_nonzero[:nij,1], 1] = tensor2[:nij, 1]
    result_tensor[edge_indices_nonzero[nij:,0], edge_indices_nonzero[nij:,1], 1] = -tensor2[:nij, 1]

    # saves memory if tensor1 and tensor2 are not loaded onto GPU before this function call
    if not device is None:
        result_tensor = result_tensor.to(device)

    result_tensor = result_tensor[:,:,[0]] + 1.j*result_tensor[:,:,[1]]
    return result_tensor

def get_vktrue_from_mol(mol, device = None, k = None):
    """Reconstruct the full rank3 self-energy tensor from node self-energy and edge_self-energy

    Args:
        tensor1 (torch.float64): the node self-energy
        tensor2 (torch.float64): the edge self-energy
        node_indices_nonzero (torch.int): indices in full nmo where graph nodes get assigned 
        edge_indices_nonzero (torch.int): indices in full tensor where edge_indices_nonzero get assigned 
        nmo (int): nmo

    Returns:
        torch.float64: cat[sigma(iw).real, sigma(iw).imag]
    """    
    if k is None:
        output = torch.zeros((mol.nkpts, mol.nmo, mol.nmo,1), dtype=torch.complex128, device=mol.nmo.device)
        for k in range(mol.nkpts):
            output[k,:,:,[0]] = construct_vk_from_graph_kpts(getattr(mol, f'vk_ii_k{k}'), getattr(mol, f'vk_ij_k{k}'), 
                        getattr(mol, f'node_indices_nonzero_k{k}'), getattr(mol, f'edge_indices_nonzero_k{k}'),
                        mol.nmo)
        return output

    return construct_vk_from_graph_kpts(getattr(mol, f'vk_ii_k{k}'), getattr(mol, f'vk_ij_k{k}'), 
                        getattr(mol, f'node_indices_nonzero_k{k}'), getattr(mol, f'edge_indices_nonzero_k{k}'),
                        mol.nmo)



def construct_vk_from_graph_polar(tensor1, tensor2, node_indices_nonzero, edge_indices_nonzero, nmo, device = None):
    """Reconstruct the full rank3 self-energy tensor from node self-energy and edge_self-energy

    Args:
        tensor1 (torch.float64): the node self-energy
        tensor2 (torch.float64): the edge self-energy
        node_indices_nonzero (torch.int): indices in full nmo where graph nodes get assigned 
        edge_indices_nonzero (torch.int): indices in full tensor where edge_indices_nonzero get assigned 
        nmo (int): nmo

    Returns:
        torch.float64: cat[sigma(iw).real, sigma(iw).imag]
    """    

    nij = edge_indices_nonzero.shape[0] // 2
    # Initialize the rank-3 tensor with zeros
    result_tensor = torch.zeros(nmo, nmo, dtype=torch.complex128, device=tensor1.device)

    # Fill the diagonal with tensor1 entries (only reals)
    result_tensor[node_indices_nonzero, node_indices_nonzero] = tensor1[:,0] + 0.0*1j # diagonal must be real
    
    result_tensor[edge_indices_nonzero[:nij,0], edge_indices_nonzero[:nij,1]] = tensor2[:nij, 0]*torch.exp(1.j*tensor2[:nij, 1])
    result_tensor[edge_indices_nonzero[nij:,0], edge_indices_nonzero[nij:,1]] = tensor2[nij:, 0]*torch.exp(-1.j*tensor2[nij:, 1])

    # saves memory if tensor1 and tensor2 are not loaded onto GPU before this function call
    if not device is None:
        result_tensor = result_tensor.to(device)

    return result_tensor


def construct_fock_from_graph_simple(tensor1, tensor2, node_indices_nonzero, edge_indices_nonzero, nmo, device = None):
    """Reconstruct the full rank3 self-energy tensor from node self-energy and edge_self-energy

    Args:
        tensor1 (torch.float64): the node self-energy
        tensor2 (torch.float64): the edge self-energy
        node_indices_nonzero (torch.int): indices in full nmo where graph nodes get assigned 
        edge_indices_nonzero (torch.int): indices in full tensor where edge_indices_nonzero get assigned 
        nmo (int): nmo

    Returns:
        torch.float64: cat[sigma(iw).real, sigma(iw).imag]
    """    

    nij = edge_indices_nonzero.shape[0] // 2
    # Initialize the rank-3 tensor with zeros
    result_tensor = torch.zeros(nmo, nmo, 2, dtype=tensor1.dtype, device=tensor1.device)

    # Fill the diagonal with tensor1 entries (only reals)
    result_tensor[node_indices_nonzero, node_indices_nonzero, 0] = tensor1[:,0]
    
    result_tensor[edge_indices_nonzero[:,0], edge_indices_nonzero[:,1],:] = tensor2

    # saves memory if tensor1 and tensor2 are not loaded onto GPU before this function call
    if not device is None:
        result_tensor = result_tensor.to(device)

    return result_tensor

def reconstruct_rank2(unraveled_array, nmo):
    # Reconstruct the N^2 x 1 array into an N x N array
    reconstructed_array = unraveled_array.view(nmo, nmo)

    return reconstructed_array

def unravel_rank2(arr):
    # Unravel the N x N array into an N^2 x 1 array
    unraveled_array = arr.flatten()

    return unraveled_array

def unravel_rank3(X):
    N, _, m = X.shape
    return X.reshape(N*N, m)   # flatten first two dims

def reconstruct_rank3(Y, N):
    m = Y.shape[-1]
    return Y.view(N, N, m)     # view is fine if contiguousen()

def cartesian_to_polar(tensor):
    """
    Converts a tensor of shape N x 2 where the first column is the real part
    and the second column is the imaginary part to a tensor where the 
    first column is the magnitude R and the second column is the phase theta.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape N x 2.
    
    Returns:
    torch.Tensor: Output tensor of the same shape N x 2, with the first
                  column as magnitude R and the second column as phase theta.
    """
    # Ensure the input is a 2D tensor with 2 columns
    if tensor.ndimension() != 2 or tensor.shape[1] != 2:
        raise ValueError("Input tensor must be of shape N x 2.")
    
    # Extract real and imaginary parts
    real = tensor[:, 0]
    imag = tensor[:, 1]
    
    # Calculate magnitude R
    R = torch.sqrt(real**2 + imag**2)
    
    # Calculate phase theta
    theta = torch.atan2(imag, real)
    
    # Stack R and theta to form the output tensor
    polar_tensor = torch.stack((R, theta), dim=1)
    
    return polar_tensor

def hermitize_feature(f):
    """
    Takes an Nk x Norb x Norb matrix and forces it to be Hermitian 
    for each k in the Nk dimension using PyTorch.
    
    Args:
    f (torch.Tensor): An Nk x Norb x Norb tensor.

    Returns:
    torch.Tensor: A Hermitian Nk x Norb x Norb tensor.
    """
    # Ensure the input is a 3-dimensional tensor
    if f.ndimension() != 3:
        raise ValueError("Input tensor must be a 3-dimensional array.")
    
    norbs = f.shape[-1]
    iu = torch.triu_indices(norbs, norbs, offset=1)  # For the assignment of full vk
    
    # Copy the upper triangular part to the lower triangular part with conjugation
    f[:, iu[1, :], iu[0, :]] = f[:, iu[0, :], iu[1, :]].conj()
    
    return f

def get_subgraph(graph, node_subset_index, modify_extras=False):
    
    x = graph.x[node_subset_index,:].clone()
    mask = (
        torch.isin(graph.edge_index[0], node_subset_index)
        & torch.isin(graph.edge_index[1], node_subset_index)
    )

    # this is tricky...
    node_mapping = -torch.ones(graph.x.size(0), dtype=torch.int64, device = graph.x.device)
    node_mapping[node_subset_index] = torch.arange(node_subset_index.size(0), device = graph.x.device)
    edge_index = graph.edge_index[:, mask]
    edge_index[0] = node_mapping[edge_index[0]]
    edge_index[1] = node_mapping[edge_index[1]]

    edge_attr = graph.edge_attr.clone()[mask,:]
    node_indices_nonzero = graph.node_indices_nonzero.clone()[node_subset_index]
    edge_indices_nonzero = graph.edge_indices_nonzero.clone()[mask,:]
    
    subgraph = graph.clone()
    subgraph.x = x
    subgraph.edge_index = edge_index
    subgraph.edge_attr = edge_attr
    subgraph.node_indices_nonzero = node_indices_nonzero
    subgraph.edge_indices_nonzero = edge_indices_nonzero

    if modify_extras:
        subgraph.dm = unravel_rank2(reconstruct_rank2(subgraph.dm, subgraph.nmo)[node_subset_index.unsqueeze(1), node_subset_index])
        subgraph.mo_energy = subgraph.mo_energy[node_subset_index]
        graph.nmo = len(node_subset_index)

    # vk_ii = graph.vk_ii.clone()[node_subset_index,:]
    # vk_ij = graph.vk_ij.clone()[mask,:]
    # subgraph.vk_ii = vk_ii
    # subgraph.vk_ij = vk_ij

    return subgraph

def sinusoid_expand(A: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    A: (N, N, d)
    L: (L,)
    Returns: (N, N, d*(L+1)) where for each channel idx_d:
        out[:,:,idx_d*(L+1) + 0]     = A[:,:,idx_d]
        out[:,:,idx_d*(L+1) + (i+1)] = sin(A[:,:,idx_d] / L[i])
    """
    N, _, d = A.shape
    Llen = L.numel()

    # A_expanded: (N, N, d, 1)
    # sin_part:   (N, N, d, L)
    sin_part = torch.sin(A.unsqueeze(-1) / L.view(1, 1, 1, -1))

    # concatenate original A along new last axis
    out = torch.cat([A.unsqueeze(-1), sin_part], dim=-1)  # (N, N, d, L+1)

    # reshape to (N, N, d*(L+1))
    out = out.reshape(N, N, d * (Llen + 1))
    return out
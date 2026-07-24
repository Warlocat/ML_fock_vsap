import torch
import torch.nn as nn
import torch.nn.functional as F

from mlvk.model.pytorch.base import TorchBase
from mlvk.model.pytorch.gnn_fast import GNN_fast
from mlvk.model.pytorch.data import get_subgraph

class GNN_Multiblock_faster(TorchBase):
    def __init__(self, **kwargs):
        super(GNN_Multiblock_faster, self).__init__(**kwargs)
        """Vk GNN
        """        
        self.scale = kwargs.get('scale', 1.0) 
        kwargs_cc = kwargs['core-core_model_kwargs']
        kwargs_vv = kwargs['valence-valence_model_kwargs']
        kwargs_cc['scale'] = self.scale
        kwargs_vv['scale'] = self.scale
        self.index_block_combinations = torch.tensor([[0, 0], [1, 1]])

        self.gnn_blocks = nn.ModuleList([GNN_fast(**kwargs_cc), GNN_fast(**kwargs_vv)])
        self.nblocks = 2
        self.n_out = self.gnn_blocks[0].n_out
        self.idx_fock = kwargs.get('idx_fock', 0) 
        

    def get_block_idx(self, inds_core, inds_val, block):
        index_tuple = (inds_core, inds_val)
        return index_tuple[self.index_block_combinations[block, 0]], index_tuple[self.index_block_combinations[block, 1]]
    
    # def get_block_pred(self, x, edge_index, edge_attr, node_idx_full, edge_index_full, nmo, nout, block):
    #     return self.gnn_blocks[block](x, edge_index, edge_attr, node_idx_full, edge_index_full, nmo, nout)
        

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_idx_full: torch.Tensor,
        edge_index_full: torch.Tensor,
        nmo: torch.Tensor,
        nout: torch.Tensor,
        inds_val: torch.Tensor,
        inds_core: torch.Tensor,
        block_output: torch.Tensor,
        this_ib = -1,
        mol = None
    ) -> torch.Tensor:
        
        if self.training:
            return self.gnn_blocks[this_ib](x, edge_index, edge_attr, node_idx_full, edge_index_full, block_output.clone())
        else:
            vk_tot = torch.zeros((nmo, nmo, nout), dtype = x.dtype, device=x.device)
            work = block_output.clone()
            for ib in range(self.nblocks):
                rows, cols = self.get_block_idx(inds_core, inds_val, ib)
                subgraph = get_subgraph(mol, rows)
                vk_ml = self.gnn_blocks[ib](subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.node_indices_nonzero, subgraph.edge_indices_nonzero, work)
                vk_tot[rows.unsqueeze(1), cols] = vk_ml[rows.unsqueeze(1), cols]
                vk_tot[cols.unsqueeze(1), rows] = vk_ml[cols.unsqueeze(1), rows]
            return vk_tot
                




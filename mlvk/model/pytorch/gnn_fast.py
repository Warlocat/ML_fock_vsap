from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch_geometric.data import Data as Graph

import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import numpy as np
import random
import argparse
import joblib
import time
import warnings
import psutil
import copy

from mlvk.model.pytorch.base import TorchBase



class MessagePassingUpdate(TorchBase):
    

    def __init__(self, **kwargs):
        
        super(MessagePassingUpdate, self).__init__(**kwargs)
        """u_e in paper
        """        
        
        self.n_input, self.n_output = kwargs.get('n_input', 20),  kwargs.get('n_output', 20)

        self.Lin = nn.Linear(self.n_input, self.n_output)
        self.act = nn.SiLU()
        
    def forward(self, x): 
        return self.act(self.Lin(x))

# second argument to AttentionalAggregation
class AttentionMessage(TorchBase):

    def __init__(self, **kwargs):
        
        super(AttentionMessage, self).__init__(**kwargs)
        """m^l_{ij} in paper
        """        
        
        self.node_attr = kwargs.get('node_attr', 20)
        self.edge_start = 2*self.node_attr
        self.n_edge_attr = kwargs.get('edge_attr', 20)
        self.n_output = kwargs.get('channels', 20)

        self.Lin1_x1 = nn.Linear(self.node_attr, self.n_edge_attr, bias = False)
        self.Lin1_x2 = nn.Linear(self.node_attr, self.n_edge_attr, bias = False)
        self.Lin2 = nn.Linear(self.n_edge_attr, self.n_output, bias = True)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        self.act_fin = nn.SiLU()
        

    def forward(self, x):
        x_edge = x[:, self.edge_start:]
        x1 = self.act1(self.Lin1_x1(x[:, :self.node_attr] + x[:, self.node_attr:self.edge_start]))
        x2 = self.act2(self.Lin1_x2(torch.abs(x[:, :self.node_attr] - x[:, self.node_attr:self.edge_start])))
        return self.act_fin(self.Lin2(x1 + x2 + x_edge))

# first argument to AttentionalAggregation
class AttentionScore(TorchBase):

    def __init__(self, **kwargs):
        
        super(AttentionScore, self).__init__(**kwargs)
        """s^l_{ij} in paper
        """       
        
        self.n_input = kwargs.get('node_attr', 20)
        self.edge_attr = kwargs.get('edge_attr', 20)
        self.n2_input = 2*self.n_input

        self.Lin_x1 = nn.Linear(self.n_input, self.edge_attr, bias = False)
        self.Lin_x2 = nn.Linear(self.n_input, self.edge_attr, bias = False)
        self.Lin_all = nn.Linear(self.edge_attr, 1, bias = False)
        self.act = nn.Tanhshrink()

    def forward(self, x):
        x_edge = x[:, self.n2_input:]
        x1 = self.Lin_x1(x[:, :self.n_input] + x[:, self.n_input:self.n2_input])
        x2 = self.Lin_x2(torch.abs(x[:, :self.n_input] - x[:, self.n_input:self.n2_input]))
        x = self.Lin_all(x1 + x2 + x_edge)
        x = self.act(x)
        return x

class Decoder(TorchBase):

    def __init__(self, **kwargs):
        
        super(Decoder, self).__init__(**kwargs)
        """decodes into Vk_{ii} or Vk_{ij}
        """        
        
        self.n_input = kwargs.get('n_input', 10)
        self.n_hidden = kwargs.get('n_hidden', 60)
        self.n_output = kwargs.get('n_output', 1)

        self.Lin1 = nn.Linear(self.n_input, self.n_hidden, bias = True)
        self.Lin2 = nn.Linear(self.n_hidden, self.n_output, bias = True)

        self.a1 = nn.SiLU()
        
    def forward(self, x):
        x = self.a1(self.Lin1(x))
        return self.Lin2(x)


class Encoder(TorchBase):

    def __init__(self, **kwargs):
        
        super(Encoder, self).__init__(**kwargs)
        """returns x^0_i in paper
        """        
        
        self.ninput = kwargs.get('ninput', 6) #20

        self.hidden_layer_width = kwargs.get('nhidden_layer', 48) #20
        self.encoding_width = kwargs.get('encoding_width', 10)

        self.Lin1 = nn.Linear(self.ninput, self.hidden_layer_width)
        self.Lin2 = nn.Linear(self.hidden_layer_width, self.hidden_layer_width, bias = True)
        self.a1 = nn.SiLU()
        self.a2 = nn.SiLU()
        self.a3 = nn.SiLU()
        self.Lin3 = nn.Linear(self.hidden_layer_width, self.encoding_width, bias = True)
        self.shortcut = nn.Linear(self.ninput, self.encoding_width, bias = True)
        
    def forward(self, x): 
        xh = self.a1(self.Lin1(x))
        xh = self.a2(self.Lin2(xh))
        xh = self.Lin3(xh) + self.shortcut(x)
        return self.a3(xh)

class LightEncoder(TorchBase):

    def __init__(self, **kwargs):
        
        super(LightEncoder, self).__init__(**kwargs)
        """returns x^0_i in paper
        """        
        
        self.ninput = kwargs.get('ninput', 6) #20

        self.hidden_layer_width = kwargs.get('nhidden_layer', 48) #20
        self.encoding_width = kwargs.get('encoding_width', 10)

        self.Lin1 = nn.Linear(self.ninput, self.hidden_layer_width)
        self.a1 = nn.SiLU()
        self.a2 = nn.SiLU()
        self.Lin2 = nn.Linear(self.hidden_layer_width, self.encoding_width, bias = True)
        self.shortcut = nn.Linear(self.ninput, self.encoding_width, bias = True)
        
    def forward(self, x): 
        xh = self.a1(self.Lin1(x))
        xh = self.Lin2(xh) + self.shortcut(x)
        return self.a2(xh)

class NodeMPL(MessagePassing):
    def __init__(self, channels,
                 aggr = 'add', **kwargs):  
        super().__init__(aggr=aggr, **kwargs)
        
        """Single message passing layer (core of the GNN implementation)

        Args:
            channels (int): N_c in paper
            aggr (str, optional OR torch_geometric aggr object): the aggregation mechanism. Defaults to 'add'.
        """      

    def message(self, x_i, x_j, edge_attr)-> Tensor:
        return torch.cat((x_i, x_j, edge_attr), dim = -1)

    def forward(self, x, edge_index, edge_attr):

        # if isinstance(x, Tensor):
        #     x: PairTensor = (x, x)
        
        node_features = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return node_features

class GNN_fast(TorchBase):
    def __init__(self, **kwargs):
        super(GNN_fast, self).__init__(**kwargs)
        """Fock GNN
        """        

        self.n_out = kwargs.get('n_out', 1) 

        # for the encoders, must align with nftr
        self.node_input = kwargs.get('node_input', 6)
        self.edge_input = kwargs.get('edge_input', 10)
        self.node_nhidden = kwargs.get('edge_nhidden', 40) 
        self.edge_nhidden = kwargs.get('edge_nhidden', 40) 
        self.scale = kwargs.get('scale', 1.0) 
        self.light = kwargs.get('light', False) 
        
        self.mpl_channels =  kwargs.get('mpl_channels', 12)
        self.message_passing_updates =  kwargs.get('message_passing_updates', 1)
        self.precision =  kwargs.get('precision', 'double')
        self.maxnmo = 100000

        if self.light:
            self.NodeEncoder = LightEncoder(encoding_width = self.mpl_channels, ninput = self.node_input, nhidden_layer= self.node_nhidden)
            self.EdgeEncoder = LightEncoder(encoding_width = self.mpl_channels, ninput = self.edge_input, nhidden_layer= self.edge_nhidden)
        else:
            self.NodeEncoder = Encoder(encoding_width = self.mpl_channels, ninput = self.node_input, nhidden_layer= self.node_nhidden)
            self.EdgeEncoder = Encoder(encoding_width = self.mpl_channels, ninput = self.edge_input, nhidden_layer= self.edge_nhidden)

        self.aggr_msgs = nn.ModuleList([AttentionMessage(node_attr = self.mpl_channels, edge_attr = self.mpl_channels, channels = self.mpl_channels) for i in range(self.message_passing_updates)])
        self.aggr_scores = nn.ModuleList([AttentionScore(node_attr = self.mpl_channels, edge_attr = self.mpl_channels) for i in range(self.message_passing_updates)])
        self.aggrs = nn.ModuleList([AttentionalAggregation(self.aggr_scores[i], self.aggr_msgs[i]) for i in range(self.message_passing_updates)])
        self.convs = nn.ModuleList([NodeMPL(channels = self.mpl_channels, aggr = self.aggrs[i]) for i in range(self.message_passing_updates)])
        self.edge_updates = nn.ModuleList([MessagePassingUpdate(n_input = self.mpl_channels, n_output = self.mpl_channels) for i in range(self.message_passing_updates)])
        self.node_updates = nn.ModuleList([MessagePassingUpdate(n_input = self.mpl_channels, n_output = self.mpl_channels) for i in range(self.message_passing_updates)])

        self.node_decoder = Decoder(n_input = self.mpl_channels, n_hidden = self.mpl_channels, n_output = self.n_out)
        self.edge_decoder = Decoder(n_input = self.mpl_channels, n_hidden = self.mpl_channels, n_output = self.n_out)

        if self.precision == 'double':
            self.double()
        if self.precision == 'float':
            self.float() 
        if self.precision == 'half':
            self.half()

    def get_er_feature(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Returns all features associated with edge (x_i, x_j, e_ij)

        Args:
            x (torch.Tensor): node attributes
            edge_index (torch.Tensor): indices of edges (2 x n_edges)
            edge_attr (torch.Tensor): edge attributes

        Returns:
            torch.Tensor: concatenation of x_i, x_j, edge_attr in order of edge_index
        """
        x_i = x.index_select(0, edge_index[0, :])
        x_j = x.index_select(0, edge_index[1, :])
        return torch.cat([x_i, x_j, edge_attr], dim=-1)

    # def construct_vk_fast(self, result_tensor: torch.Tensor, vk_ii: torch.Tensor, vk_ij: torch.Tensor, 
    #                       node_idx_full:torch.Tensor, edge_index_full:torch.Tensor, 
    #                       nmo:int, nout:int) -> torch.Tensor:
    #     """Reconstruct the full output from node values and edge values

    #     Args:
    #         tensor1 (torch.float64): the node self-energy
    #         tensor2 (torch.float64): the edge self-energy
    #         graph (gnn_orchestrator.GraphData object): the graph object that stores the indices of nodes and edges that have not been removed from the DFT graph

    #     Returns:
    #         torch.float64: vk
    #     """    
        

    #     # Fill the diagonal with tensor1 entries
    #     result_tensor[node_idx_full, node_idx_full,:] = vk_ii

    #     # Fill the upper and lower triangle with tensor2 entries (graph is undirected so we don't have to flip stuff, though it could save computation)
    #     result_tensor[edge_index_full[:,0], edge_index_full[:,1],:] = vk_ij

    #     return result_tensor

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_idx_full: torch.Tensor,
        edge_index_full: torch.Tensor,
        result_tensor: torch.Tensor
    ):
        x_tmp = self.NodeEncoder(x)
        e_tmp = self.EdgeEncoder(edge_attr)
        sum_x = x_tmp
        sum_e = e_tmp
        for i, (edge_update, aggr_msg, conv, node_update) in enumerate(zip(
                self.edge_updates, self.aggr_msgs, self.convs, self.node_updates)):
            e_new = self.get_er_feature(x_tmp, edge_index, e_tmp)
            e_new = edge_update(aggr_msg(e_new))
            x_new = conv(x_tmp, edge_index, e_tmp)
            x_new = node_update(x_new)
            # Append to hidden states
            sum_x = sum_x + x_new
            sum_e = sum_e + e_new
            x_tmp = x_new
            e_tmp = e_new

        # torch._dynamo.mark_dynamic(result_tensor, 3, min = 1, max = self.maxnmo)
        vk_ii, vk_ij = self.node_decoder(sum_x), self.edge_decoder(sum_e)
        result_tensor[node_idx_full, node_idx_full,:] = vk_ii
        result_tensor[edge_index_full[:,0], edge_index_full[:,1],:] = vk_ij

        # vk = self.construct_vk_fast(, , node_idx_full, edge_index_full, nmo, nout)

        if self.training:
            return result_tensor
        else:
            return result_tensor*self.scale
       

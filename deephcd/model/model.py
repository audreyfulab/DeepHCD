import torch
import torch.nn as nn
import torch.nn.functional as F
from deephcd.model.layer import Fully_ConnectedLayer, Comm_DenseLayer2, AE_layer
import torch_geometric.utils as pyg_utils
from collections import OrderedDict
from torchinfo import summary
from torch_kmeans import SoftKMeans
from typing import Optional, Union, List,  Literal
import numpy as np
import os
import time

# Accumulated timing across all forward() calls (reset by training script between epochs)
forward_timing = {
    # forward pass breakdown
    'total_forward': 0.0,
    'input_norm': 0.0,
    'edge_index': 0.0,
    'gate_encoder': 0.0,
    'dot_product': 0.0,
    'gate_decoder': 0.0,
    'output_layers': 0.0,
    'top_comm': 0.0,
    'select_subsets': 0.0,
    'middle_comm': 0.0,
    'clustering': 0.0,
    # per-batch training overhead
    'batch_prep': 0.0,
    'loss_compute': 0.0,
    'backward': 0.0,
    'grad_clip': 0.0,
    'optimizer_step': 0.0,
    'gpu_cleanup': 0.0,
    # epoch-level overhead
    'validation': 0.0,
    'perf_eval': 0.0,
    'calls': 0,
}

def reset_forward_timing():
    """Reset accumulated forward-pass timing stats."""
    for key in forward_timing:
        forward_timing[key] = 0.0 if key != 'calls' else 0


def select_class(X: torch.Tensor, labels: torch.Tensor, k: int, dim: int = 0, return_index: bool = False):
    #L = torch.tensor(labels)
    indices = torch.nonzero(labels == k)
    X_sub = torch.index_select(X, dim=dim, index=indices.squeeze())
    
    if return_index:
        return X_sub, indices
    else:
        return X_sub


def select_subgraph(A: torch.Tensor, labels: torch.Tensor, k: int):
    #L = torch.tensor(labels)
    indices = torch.nonzero(labels == k).squeeze()
    A_rows = torch.index_select(A, dim=0, index=indices)
    # Select the rows corresponding to community 1
    subgraph = torch.index_select(A_rows, dim = 1, index = indices)
    return subgraph


def reorganize_labels(S1: torch.Tensor, S2_list: torch.Tensor):
    # S1: tensor of initial class labels
    # S2_list: list of tensors, each containing predicted labels for subsets of X based on unique labels in S1

    # Create a list to hold the indices for each unique label in S1
    indices_list = [torch.where(S1 == label)[0] for label in torch.unique(S1, sorted=False)]

    # Concatenate S2 tensors according to the order of indices
    S2_reorganized = torch.empty_like(S1)
    for indices, S2 in zip(indices_list, S2_list):
        S2_reorganized[indices] = S2

    return S2_reorganized
    


class GATE(nn.Module):
    """
    GATE model described in https://arxiv.org/pdf/1905.10715.pdf
    
    """

    def __init__(self, in_nodes: int, in_attrib: int, normalize: bool = True, hid_sizes: List[int] = [256, 128, 64], 
                 attn_heads: int = 1, layer_act: nn.Module = nn.Identity(), dropout: float = 0.2, 
                 operator: Literal['GATConv', 'GATv2Conv', 'SAGEConv'] = 'GATv2Conv', **kwargs):
        
        super(GATE, self).__init__()
        #store size
        self.in_nodes = in_nodes
        self.in_attrib = in_attrib
        
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        for idx, out in enumerate(hid_sizes):
            
            # add multi head attendtion layers to dictionary
            layer_name = f'{operator}_'+str(idx)+'-'+str(out)
            module_dict.update({layer_name: AE_layer(nodes = in_nodes, 
                                                      in_features=in_attrib,
                                                      out_features=out,
                                                      heads=attn_heads,
                                                      norm = normalize,
                                                      operator = operator,
                                                      dropout = dropout,
                                                      **kwargs)})
        
            
            module_dict.update({'act'+str(out): layer_act})
            
            in_attrib = out
        
        #build model by pytorch sequential
        self.seqmodel = nn.Sequential(module_dict)
        
        
    def forward(self, X, A, ei=None, ea=None):
        weights_list = []
        if ei is None:
            ei, ea = pyg_utils.dense_to_sparse(A)
        H, E, attr, weights_list = self.seqmodel((X, ei, ea, weights_list))

        return (H, A, weights_list)
    

    


class AddLearningLayers(nn.Module):
    """
    extra layers between embedding and comm prediction layers that aim
    to improve class learning
    """
    def __init__(self, in_nodes: int, in_attrib: int, sizes: List[int] = [64, 32], 
                 normalize: bool = True, dropout: float = 0.2, negative_slope: float = 0.2):
        super(AddLearningLayers, self).__init__()
        self.nodes = in_nodes
        self.attrib = in_attrib
                
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        
        for idx, size in enumerate(sizes):
            #add output layers to dictionary
            module_dict.update({f'LinearLayer_{size}_{idx}': Fully_ConnectedLayer(
                in_features = in_attrib, 
                out_features = size,
                norm = normalize,
                dropout = dropout,
                alpha=negative_slope
                )})
            
            in_attrib = size
     
        #build model by pytorch sequential
        self.model = nn.Sequential(module_dict)
        
    def forward(self, Z):
        
        H = self.model(Z)
        
        return H



class CommunityDetectionLayers(nn.Module):
    """
    Community Detection Module
    
    """

    def __init__(self, in_nodes, in_attrib, comm_sizes=[60, 10], 
                 layer_operator = 'Linear', dropout = 0.2, normalize = True, 
                 input_transform_layer = False, **kwargs):
        
        super(CommunityDetectionLayers, self).__init__()
        #store size
        self.in_nodes = in_nodes
        self.in_attrib = in_attrib
        #create empty ordered dictionary for pytorch sequential model build
        module_dict = OrderedDict([])
        
        for idx, comms in enumerate(comm_sizes):
            #add output layers to dictionary
            module_dict.update({f'Comm_{layer_operator}_'+str(idx): Comm_DenseLayer2(
                in_features = in_attrib, 
                out_comms = comms,
                norm = normalize,
                dropout = dropout,
                operator = layer_operator,
                **kwargs
                )})
        
        #build model by pytorch sequential
        self.model = nn.Sequential(module_dict)
        
        
    def forward(self, Z, A, ei=None, ea=None):
        inputs = [Z, A, [], [], [], []]
        if ei is not None:
            inputs.append(ei)
        H_layers = self.model(inputs)

        return H_layers[:6]
        


class HCD(nn.Module):
    """
    Hierarchical Graph Representation Network for genes
    nodes: (integer) number of nodes in attributed graph
    attrib: (integer) number of node-attributes (i.e features)
    hidden_dims: (list) of integers giving the size of the hidden layers
    comm_sizes: (list) giving the number of super nodes/communities in 
                hierarchcial layers
    **kwargs: Keyword arguments passed to GATE/GAT module
    """

    def __init__(self, nodes, attrib, ae_hidden_dims = [256, 128, 64], method = ['top_down', 'bottom_up'],
                 ll_hidden_dims = [64, 64], comm_sizes = [60, 10], ae_operator = 'GATv2Conv',
                 use_kmeans_top = False, use_kmeans_middle = False, comm_operator = 'Linear', dropout = 0.2, 
                 use_output_layers = False, normalize_outputs = False, normalize_input = False, ae_attn_heads=1, **kwargs):
        
        super(HCD, self).__init__()
        #copy and reverse decoder layer dims
        decode_dims = ae_hidden_dims.copy()
        decode_dims.reverse()
        decode_dims.append(attrib)
        self.method = method
        self.use_output_layers = use_output_layers
        self.use_kmeans_top = use_kmeans_top
        self.use_kmeans_middle = use_kmeans_middle
        self.comm_sizes = comm_sizes
        self.ae_hidden_dims = ae_hidden_dims
        self.ll_hidden_dims = ll_hidden_dims
        self.normalize_outputs = normalize_outputs
        self.comm_operator = comm_operator
        self.ae_operator = ae_operator
        self.dropout_rate = dropout
        
        #GATE
        #set up encoder
        self.encoder = GATE(in_nodes = nodes, 
                            in_attrib = attrib, 
                            hid_sizes=ae_hidden_dims, 
                            normalize = self.normalize_outputs, 
                            operator= self.ae_operator,
                            attn_heads = ae_attn_heads,
                            dropout = self.dropout_rate)
        #set up decoder
        self.decoder = GATE(in_nodes = nodes, 
                            in_attrib = self.ae_hidden_dims[-1], 
                            hid_sizes=decode_dims[1:], 
                            normalize = self.normalize_outputs, 
                            operator = self.ae_operator,
                            attn_heads = ae_attn_heads,
                            dropout = self.dropout_rate)
        
        #bottom up method
        if self.method == 'bottom_up':
            if self.use_output_layers:
                #extra MLP layers between embedding and community detection step
                self.fully_connected_layers = AddLearningLayers(in_nodes=nodes, 
                                                                in_attrib=self.ae_hidden_dims[-1],
                                                                sizes=self.ll_hidden_dims,
                                                                normalize=self.normalize_outputs,
                                                                dropout = self.dropout_rate)
            
            
                #set up community detection module
                self.commModule = CommunityDetectionLayers(in_nodes = nodes, 
                                                            in_attrib = self.ll_hidden_dims[-1], 
                                                            normalize = self.normalize_outputs, 
                                                            comm_sizes = self.comm_sizes,
                                                            layer_operator = self.comm_operator,
                                                            dropout = self.dropout_rate,
                                                            **kwargs)
            else:
                #set up community detection module
                self.commModule = CommunityDetectionLayers(in_nodes = nodes, 
                                                            in_attrib = self.ae_hidden_dims[-1], 
                                                            normalize = self.normalize_outputs, 
                                                            comm_sizes=self.comm_sizes,
                                                            layer_operator = self.comm_operator,
                                                            dropout = self.dropout_rate,
                                                            **kwargs)
                
        #Top down method 
        elif self.method == 'top_down':
            self.comm_sizes = comm_sizes[::-1]
            
            if self.use_output_layers:
                self.fully_connected_layers = AddLearningLayers(in_nodes=nodes, 
                                                                in_attrib=self.ae_hidden_dims[-1],
                                                                sizes=self.ll_hidden_dims,
                                                                normalize=self.normalize_outputs,
                                                                dropout = self.dropout_rate)
                comm_in_dim = self.ll_hidden_dims[-1]
            else:
                comm_in_dim = self.ae_hidden_dims[-1]
                
            if self.use_kmeans_top:
                self.TopCommModule = SoftKMeans(n_clusters=self.comm_sizes[0], max_iter=1000, num_init=10,
                                                init_method='k-means++', verbose=False)
            
            else:
                self.TopCommModule = CommunityDetectionLayers(in_nodes = nodes, 
                                                              in_attrib = comm_in_dim, 
                                                              normalize = self.normalize_outputs, 
                                                              comm_sizes = [self.comm_sizes[0]],
                                                              layer_operator = self.comm_operator,
                                                              dropout = self.dropout_rate,
                                                              **kwargs)
            if len(self.comm_sizes) > 1:
                if self.use_kmeans_middle:
                    self.MiddleModules = [SoftKMeans(n_clusters=self.comm_sizes[1], 
                                                     max_iter=1000, 
                                                     num_init=10) for i in enumerate(range(0, self.comm_sizes[0]))]
                else:
                    #separate layers for each partition in top
                    self.MiddleModules = [CommunityDetectionLayers(in_nodes = nodes, 
                                                                   in_attrib = comm_in_dim, 
                                                                   normalize = self.normalize_outputs, 
                                                                   comm_sizes = [self.comm_sizes[1]],
                                                                   layer_operator = self.comm_operator,
                                                                   dropout = self.dropout_rate,
                                                                   **kwargs) for i in range(0, self.comm_sizes[0])]
            
            
            
        else:
            print('ERROR: method not specified!')
            
        
        if normalize_input:
            self.input_norm = nn.LayerNorm(attrib)
        else:
            self.input_norm = nn.Identity()
        
        #set dot product decoder activation to sigmoid
        self.dpd_act = nn.Sigmoid()
        #normalization for dpd_activation
        self.dpd_norm = nn.Identity()
        
        
    def forward(self, X, A, ei=None, ea=None):
        _t_forward = time.perf_counter()
        device = X.device

        self.to(device)

        if hasattr(self.input_norm, 'weight') and self.input_norm.weight.device != X.device:
           self.input_norm = self.input_norm.to(X.device)

        # normalize input
        _t0 = time.perf_counter()
        H = self.input_norm(X)
        forward_timing['input_norm'] += time.perf_counter() - _t0

        # Pre-compute sparse edge_index once for the full batch A so encoder,
        # decoder, and top community module all reuse it without re-scanning A.
        if ei is None:
            _t0 = time.perf_counter()
            ei, ea = pyg_utils.dense_to_sparse(A)
            forward_timing['edge_index'] += time.perf_counter() - _t0

        # get embedding representation
        _t0 = time.perf_counter()
        Z, A, encoder_attention_weights = self.encoder(H, A, ei=ei, ea=ea)
        forward_timing['gate_encoder'] += time.perf_counter() - _t0

        # Normalize embeddings and compute dot-product reconstruction
        _t0 = time.perf_counter()
        Z_norm = F.normalize(Z, p=2, dim=1)
        sim = torch.mm(Z_norm, Z_norm.T)
        A_hat = self.dpd_act(sim)
        sim = torch.clamp(sim, -10, 10)
        A_logits = self.dpd_norm(torch.mm(Z, Z.transpose(0,1)))
        forward_timing['dot_product'] += time.perf_counter() - _t0

        # get reconstructed adjacency
        _t0 = time.perf_counter()
        X_hat, A, decoder_attention_weights = self.decoder(Z, A, ei=ei, ea=ea)
        forward_timing['gate_decoder'] += time.perf_counter() - _t0

        _t_clust = time.perf_counter()
        # bottom up method
        if self.method == 'bottom_up':
            subsets_X = []
            subsets_A = []
            if self.use_output_layers:
                # Output learning layers
                _t0 = time.perf_counter()
                W = self.fully_connected_layers(Z)
                forward_timing['output_layers'] += time.perf_counter() - _t0

                # fit hierarchy
                _t0 = time.perf_counter()
                X_top, A_top, X_all, A_all, P_all, S_all = self.commModule(W, A, ei=ei, ea=ea)
                forward_timing['top_comm'] += time.perf_counter() - _t0
            else:
                _t0 = time.perf_counter()
                X_top, A_top, X_all, A_all, P_all, S_all = self.commModule(Z, A, ei=ei, ea=ea)
                forward_timing['top_comm'] += time.perf_counter() - _t0

        # top down method
        if self.method == 'top_down':
                # fit hierarchy

                # Get initial set of labels S - a list with one element (a tensor of class labels)
                if self.use_kmeans_top:
                    if self.use_output_layers:
                        _t0 = time.perf_counter()
                        W = self.fully_connected_layers(Z)
                        forward_timing['output_layers'] += time.perf_counter() - _t0
                        _t0 = time.perf_counter()
                        result = self.TopCommModule(W.unsqueeze(0))
                        forward_timing['top_comm'] += time.perf_counter() - _t0
                    else:
                        _t0 = time.perf_counter()
                        result = self.TopCommModule(Z.unsqueeze(0))
                        forward_timing['top_comm'] += time.perf_counter() - _t0
                    S = [result.labels.squeeze(0)]
                    P = result.soft_assignment.squeeze(0)
                else:
                    if self.use_output_layers:
                        # Output learning layers
                        _t0 = time.perf_counter()
                        W = self.fully_connected_layers(Z)
                        forward_timing['output_layers'] += time.perf_counter() - _t0
                        _t0 = time.perf_counter()
                        X_top, A_top, X_all, A_all, P_all, S = self.TopCommModule(W, A, ei=ei, ea=ea)
                        forward_timing['top_comm'] += time.perf_counter() - _t0
                    else:
                        _t0 = time.perf_counter()
                        X_top, A_top, X_all, A_all, P_all, S = self.TopCommModule(Z, A, ei=ei, ea=ea)
                        forward_timing['top_comm'] += time.perf_counter() - _t0

                    P = P_all[0]


                # Select data based on top partition i.e S
                _t0 = time.perf_counter()
                if self.use_output_layers:
                    subsets_with_index = [select_class(W, S[0], k, dim=0, return_index=True) for k in torch.unique(S[0])]
                else:
                    subsets_with_index = [select_class(Z, S[0], k, dim=0, return_index=True) for k in torch.unique(S[0])]
                subsets_Z = [i[0] for i in subsets_with_index]
                subsets_X = [select_class(X, S[0], k, dim=0) for k in torch.unique(S[0])]
                subsets_A = [select_subgraph(A, S[0], k) for k in torch.unique(S[0])]
                forward_timing['select_subsets'] += time.perf_counter() - _t0

                if len(self.comm_sizes) > 1:
                    if self.use_kmeans_middle:

                        # apply k softkmeans layers
                        _t0 = time.perf_counter()
                        results = [self.MiddleModules[i](x = sub_Z.unsqueeze(0), k = min(sub_Z.shape[0], self.comm_sizes[1])) for idx, (i, sub_Z) in enumerate(zip(torch.unique(S[0]), subsets_Z)) if sub_Z.shape[0] > 1]
                        forward_timing['middle_comm'] += time.perf_counter() - _t0

                        # store results
                        X_all = []
                        A_all = []
                        P_all = [P, [i.soft_assignment.squeeze(0) for i in results]]
                        S_temp = [i.labels.squeeze(0)+index*j for index, (i,j) in enumerate(zip(results, torch.arange(self.comm_sizes[0])[torch.unique(S[0])]))]
                        S_final = reorganize_labels(S1 = S[0], S2_list= S_temp)
                        S_all = [S[0], S_final]


                    else:
                        device = X.device
                        for m in self.MiddleModules:
                           m.to(device)
                        device = Z.device
                        # apply k linear predictors
                        _t0 = time.perf_counter()
                        results = [self.MiddleModules[i.item()](sub_Z.to(device), sub_A.to(device)) for idx, (i, sub_Z, sub_A) in enumerate(zip(torch.unique(S[0]), subsets_Z, subsets_A))]
                        forward_timing['middle_comm'] += time.perf_counter() - _t0

                        # store results
                        X_all = [i[0] for i in results]
                        A_all = [i[1] for i in results]
                        P_all = [P, [i[4][0] for i in results]]
                        S_temp = [i[5][0]+((self.comm_sizes[1]+10)*index) for index, i in enumerate(results) ]
                        S_final = reorganize_labels(S1 = S[0], S2_list= S_temp)
                        S_all = [S[0], S_final]

        forward_timing['clustering'] += time.perf_counter() - _t_clust
        forward_timing['total_forward'] += time.perf_counter() - _t_forward
        forward_timing['calls'] += 1

        A_all_final = [A]+[A_all]+[subsets_A]
        X_all_final = [Z]+[X_all]+[subsets_X]
        return X_hat, A_hat, A_logits, X_all_final, A_all_final, P_all, S_all, {'encoder': [[i.cpu() for i in j] for j in encoder_attention_weights], 'decoder': [[i.cpu() for i in j] for j in decoder_attention_weights]}
    
    
    
    
    
    def summarize(self):
        print('-----------------GATE-Encoder-------------------')
        summary(self.encoder)
        print('-----------------GATE-Decoder-------------------')
        summary(self.decoder)
        if self.use_output_layers:
            print('------------Fully-Connected-Layers--------------')
            summary(self.fully_connected_layers)
        print('----------Community-Detection-Module------------')
        
        print(f'METHOD: {self.method}')
        if self.use_kmeans_top:
            print(f'KMEANS -- TOP: {self.use_kmeans_top}') 
        if self.use_kmeans_middle:
            print('MIDDLE: {self.use_kmeans_middle}')
        if self.method == 'bottom_up':
            summary(self.commModule)
        else:
            if not self.use_kmeans_top:
                print('TOP LAYER: \n')
                summary(self.TopCommModule)
            if not self.use_kmeans_middle:
                print('MIDDLE LAYERS: \n')
                for i in range(self.comm_sizes[0]):
                    print(f'COMMUNITY {i} MODEL: \n')
                    summary(self.MiddleModules[i])

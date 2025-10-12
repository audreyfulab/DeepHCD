import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv, GraphNorm
import torch_geometric.utils as pyg_utils
#mport seaborn as sbn
#import matplotlib.pyplot as plt
from typing import Optional, Union, List,  Literal

#layer after updating using llama3
class AE_layer(nn.Module):
    """
    GAT layer described in https://arxiv.org/pdf/1905.10715.pdf
    Gain Setting Recommendations for Xavier Initialization
        activation       recommended
        function         gain
        =======================================
        sigmoid()        1.0
        tanh()           5/3 = 1.6667
        relu()           sqrt(2) = 1.4142
        Identity         1.0
        Convolution      1.0
        LeakyReLU        sqrt(2 / (1 + (-m)^2)
    """

    def __init__(self, nodes: int, in_features: int, out_features: int, act: nn.Module = nn.Identity(), 
                 norm: bool = True, heads: int = 1, dropout: float = 0.2, operator: Literal['GATConv', 'GATv2Conv', 'SAGEConv'] = 'GATv2Conv'):
        super(AE_layer, self).__init__()
        # Store in and features for layer
        self.nodes = nodes
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.norm = norm
        self.operator = operator

        # Define the GAT layer
        if self.operator ==  'GATConv':
            
            self.layer = GATConv(in_channels = in_features, 
                                 out_channels = out_features,
                                 heads=heads,
                                 dropout=dropout,
                                 concat = False)
            
        if self.operator == 'GATv2Conv':
            
            self.layer = GATv2Conv(in_channels=in_features,
                                   out_channels=out_features,
                                   heads=heads,
                                   dropout=dropout,
                                   concat = False)
            
        if self.operator == 'SAGEConv':
            
            self.layer = SAGEConv(in_channels = in_features, 
                                  out_channels = out_features,
                                  aggr='mean',
                                  normalize=norm)
        
        # Define normalization layer
        if self.norm:
            #self.act_norm = nn.LayerNorm(out_features)
            self.act_norm = GraphNorm(out_features)
        else:
            self.act_norm = nn.Identity()
        
        #self.final_linear = nn.Linear(out_features * heads, out_features, bias = use_bias)
        
    def forward(self, inputs):
        """
        Forward pass for the GAT layer.
        """
        X, E, attr, attention_list = inputs
        
        if self.operator == 'SAGEConv':
            M = self.layer(x=X, edge_index=E)
            attention_list = []
        else:
            # Compute the GAT layer output
            H, (edge_index, attention_weights) = self.layer(x=X, edge_index=E, return_attention_weights=True)
        
            # append attention weights to existing list
            attention_list.append((edge_index, attention_weights))
            # Apply normalization
            M = self.act_norm(H)

        return (M, E, attr, attention_list)







class Fully_ConnectedLayer(nn.Module):
    """
    
    """

    def __init__(self, in_features: int, out_features: int, norm: bool = True, alpha: float = 0.2, 
                 use_bias: bool = True, dropout: float = 0.2):

        super(Fully_ConnectedLayer, self).__init__()
        #store community info
        self.in_features = in_features
        self.out_features = out_features
        self.norm = norm
            
        self.Linearlayer = nn.Linear(in_features = self.in_features, 
                                     out_features = self.out_features, 
                                     bias = use_bias)
        
        self.Dropout_layer = nn.Dropout1d(p = dropout)
        #set layer activation
        self.act = nn.LeakyReLU(negative_slope=alpha)
        #normalize inputs
        if self.norm == True:
            self.act_norm = nn.BatchNorm1d(self.out_features)
        else:
            self.act_norm = nn.Identity()
            
            
    def forward(self, X):
        
        #apply linear layer and dropout
        H = self.Linearlayer(X)
        
        #normalize and return
        H_norm = self.act_norm(H)
        
        #apply LeakyReLU activation
        H_act = self.act(H_norm)
        
        #apply dropout
        H_out = self.Dropout_layer(H_act)
        
        return H_out




class Comm_DenseLayer2(nn.Module):
    """
    A dense layer implementation for hierarchical community detection in graphs using various graph neural network operators.
    This layer transforms node representations and produces community assignments using different graph neural network
    architectures (Linear, GAT, GATv2, or GraphSAGE).

    Parameters
    ----------
    in_features : int
        Number of input features for each node
    out_comms : int
        Number of output communities (classes) to detect
    norm : bool, default=True
        Whether to apply normalization to layer outputs. Uses GraphNorm for graph-based operators 
        and LayerNorm for Linear operator
    alpha : float, default=0.2
        Negative slope coefficient for LeakyReLU activation and GAT attention mechanism
    use_bias : bool, default=True
        Whether to include bias terms in linear transformations
    dropout : float, default=0.2
        Dropout rate for GAT attention mechanisms
    heads : int, default=1
        Number of attention heads for GAT layers
    operator : str, default=['Linear', 'Conv2d', 'GATConv', 'GATv2Conv', 'SAGEConv']
        The type of graph neural network operator to use for node transformations

    Attributes
    ----------
    transform : torch.nn.Module
        The main transformation layer (Linear, Conv1d, GAT, GATv2, or GraphSAGE)
    output_linear : torch.nn.Linear
        Final linear layer for community assignment
    out_norm : torch.nn.Module
        Normalization layer (GraphNorm, LayerNorm, or Identity)
    act : torch.nn.LeakyReLU
        Activation function

    Methods
    -------
    forward(inputs)
        Processes the input node representations and graph structure to produce community assignments
        and updated graph representations.
        
        Args:
            inputs (list): 5 elements Contains [Z, A, X_tilde, A_tilde, S] where:
                - Z (torch.Tensor): Node representations (N x q)
                - A (torch.Tensor): Adjacency matrix (N x N)
                - X_tilde: Centroid matrix   (ki x q) can be an empty list
                - A_tilde (torch.Tensor): Graph for connected communities (ki x ki) - can be an empty list
                - S (torch.Tensor): Predicted class labels (N x 1) - can be an empty list
        
        Returns:
            list: Updated inputs containing:
                - Z: Updated node representations
                - A: Updated community adjacency matrix
                - Centroids: List of community centroids
                - Subgraphs: List of community subgraphs
                - Assignment probabilities: Soft community assignments
                - Node labels: Hard community assignments

    Notes
    -----
    The layer supports multiple graph neural network operators with different characteristics:
    - Linear: Basic linear transformation
    - Conv2d: Basic 2D Convolution
    - SAGEConv: GraphSAGE convolution for neighborhood aggregation
    - GATConv: Graph Attention Network for attention-based message passing
    - GATv2Conv: Improved Graph Attention Network with dynamic attention
    
    The forward pass includes:
    1. Node feature transformation using the selected operator
    2. Normalization and activation
    3. Community assignment probability computation
    4. Community centroid and adjacency matrix updates
    5. Storage of intermediate representations and assignments
    """

    def __init__(self, in_features: int, out_comms: int, norm: bool = True, alpha: float = 0.2, use_bias: bool = True, 
                 dropout: float = 0.2, heads: int = 1, operator: Literal['None', 'Linear', 'Conv1d', 'GATConv', 'GATv2Conv', 'SAGEConv'] = 'Linear',
                 kernal_size: int = 4, stride: int = 2, padding: int = 0, **kwargs):
        
        super(Comm_DenseLayer2, self).__init__()
        #store community info
        self.in_features = in_features
        self.out_features = out_comms
        self.out_comms = out_comms
        self.norm = norm
        self.operator = operator
        
        
        if self.operator.lower() == 'none':
            
            self.transform = nn.Identity()
        
        elif self.operator.lower() == 'linear':
            
            self.transform = nn.Linear(in_features = self.in_features, 
                                   out_features = self.in_features, 
                                   bias = use_bias)
            
        elif self.operator.lower() == 'conv1d':
        
            self.transform = nn.Conv1d(in_channels = self.in_features, 
                                       out_channels = self.in_features,
                                       kernel_size = kernal_size,
                                       stride=stride, 
                                       padding=padding,
                                       **kwargs)   
            
        elif self.operator.lower() == 'sageconv':
            
            self.transform = SAGEConv(in_channels = self.in_features, 
                                  out_channels = self.in_features,
                                  aggr='mean',
                                  normalize=norm)
            
        elif self.operator.lower() == 'gatconv':
            
            self.transform = GATConv(in_channels = self.in_features, 
                                 out_channels = self.in_features,
                                 heads=heads,
                                 negative_slope=alpha,
                                 dropout=dropout,
                                 concat = False)
        elif self.operator.lower() == 'gatv2conv':
            
            self.transform = GATv2Conv(in_channels=self.in_features,
                                   out_channels=self.in_features,
                                   heads=heads,
                                   negative_slope=alpha,
                                   dropout=dropout,
                                   concat = False)
            
        else:
            raise ValueError(f'ERROR - "{operator}" is invalid for argument "operator"')
            
        self.output_linear = nn.Linear(in_features = self.in_features, 
                                   out_features = self.out_features, 
                                   bias = use_bias)
        
        #normalize layer output
        if self.norm == True:
            if self.operator in ['GATConv', 'GraphSAGE', 'GATv2Conv']:
                self.out_norm = GraphNorm(self.in_features)
            else:
                self.out_norm = nn.LayerNorm(self.in_features)
                #self.out_norm = nn.LayerNorm(self.out_features)
        else:
            self.out_norm = nn.Identity()
            
        #set layer activation
        self.act = nn.LeakyReLU(negative_slope=alpha)

    
    def forward(self, inputs):
        """
        inputs: a list of size elements [Z, A, X_tilde, A_tilde, S] where 
                Z: Node representations   N x q
                A: Adjacency matrix   N x N
                X_tilde: Centroid matrix   ki x q
                A_tilde: graph for connected communities   ki x ki
                S: predicted class labels   N x 1
        """
        Z=inputs[0]
        A=inputs[1]
        
        if self.operator.lower() == 'none':
            
            H = self.transform(Z)
        
        if self.operator.lower() in ['linear','conv1d']:
            
            #linear layer and activation
            M = self.transform(Z)
            M_norm = self.out_norm(M)
            H = self.act(M_norm)
            
        if self.operator.lower() in ['sageconv','gatconv','gatv2conv']:
            
            ei, ea = pyg_utils.dense_to_sparse(A)
            M = self.transform(x=Z, edge_index=ei)
            H = self.out_norm(M)
        
        # class prediction probabilities
        OL = self.output_linear(H)
        P = F.softmax(OL, dim = 1)
        
        #get the centroids and layer adjacency matrix
        X_tilde = torch.mm(torch.mm(Z.T, P), torch.diag(1/P.sum(dim = 0)+1e-8)).T
        
        #X_tilde = self.act(torch.mm(Z.transpose(0,1), P))
        A_tilde = torch.mm(P.transpose(0,1), torch.mm(A, P))
        
        #store
        inputs[0] = X_tilde
        inputs[1] = A_tilde
        #get assignment labels
        S = torch.argmax(P, dim = 1)
        #store subgraphs, assignment probs, and node labels
        inputs[2].append(X_tilde)
        inputs[3].append(A_tilde)
        inputs[4].append(P)
        inputs[5].append(S)
        
        return inputs
     
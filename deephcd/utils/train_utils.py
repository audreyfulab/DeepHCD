import torch
import torch.nn as nn
from typing import Optional, List, Literal, Any, Tuple
import numpy as np
from deephcd.utils.utilities import get_layered_performance, trace_comms
import gc
from contextlib import contextmanager

#====================================================================================
# context manager
#====================================================================================
@contextmanager
def memory_efficient_context():
    """
    Context manager for aggressive memory cleanup.

    Ensures CPU and GPU memory are released after the enclosed block executes.
    Useful in long-running loops or between large model operations.

    Yields
    ------
    None
        Executes the enclosed code block.
    """
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

#====================================================================================
# data mini batching utility that batches on features
#====================================================================================
def get_efficient_batches(
    X: torch.Tensor,
    batch_size: Optional[int] = 64,
    device: Optional[Literal["cpu", "cuda"]] = "cpu"
) -> list[torch.Tensor]:
    """
    Generate memory-efficient mini-batches for graph-structured data.

    This function creates randomized batches of node indices for training 
    large-scale graph data in limited memory environments. The batching is 
    performed using index permutation rather than full tensor slicing to 
    minimize GPU memory usage.

    Parameters
    ----------
    X : torch.Tensor
        Node feature matrix of shape `(num_nodes, num_features)`.
    batch_size : int, default=64
        Number of nodes per batch.
    device : {'cpu', 'cuda'}, default='cpu'
        Device where the random permutation of indices is generated.

    Returns
    -------
    list[torch.Tensor]
        A list of 1D tensors, each containing the node indices for one batch.

    Examples
    --------
    >>> X = torch.randn(1000, 64)
    >>> A = torch.eye(1000)
    >>> batches = get_efficient_batches(X, batch_size=128)
    >>> len(batches)
    8
    >>> batches[0].shape
    torch.Size([128])
    """
    num_nodes = X.size(0)
    indices = torch.randperm(num_nodes, device=device)
    
    batches = []
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        batch_indices = indices[start:end]
        batches.append(batch_indices)
    
    return batches

#====================================================================================
# retreives mini-batch data based on precomputed batch indices
#====================================================================================
def get_batch_data(
    X: torch.Tensor,
    A: torch.Tensor,
    batch_indices: torch.Tensor,
    device: Optional[Literal["cpu", "cuda"]] = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract a batch of node features and adjacency submatrix in a memory-efficient way.

    This function dynamically slices node feature and adjacency matrices 
    based on provided batch indices, moving only the required data to the 
    specified device. This helps reduce GPU memory overhead during training 
    on large graphs.

    Parameters
    ----------
    X : torch.Tensor
        Node feature matrix of shape `(num_nodes, num_features)`.
    A : torch.Tensor
        Full adjacency matrix of shape `(num_nodes, num_nodes)`.
    batch_indices : torch.Tensor
        1D tensor of node indices representing the current batch.
    device : {'cpu', 'cuda'}, default='cpu'
        Target device to move the batch tensors to.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        A tuple `(X_batch, A_batch)` where:
        - `X_batch` is the sliced feature matrix for the batch.
        - `A_batch` is the corresponding adjacency submatrix.

    Examples
    --------
    >>> X = torch.randn(1000, 64)
    >>> A = torch.eye(1000)
    >>> batch = torch.arange(0, 128)
    >>> Xb, Ab = get_batch_data(X, A, batch, device="cuda")
    >>> Xb.shape, Ab.shape
    (torch.Size([128, 64]), torch.Size([128, 128]))
    """
    X_batch = X[batch_indices].to(device)
    A_batch = A[batch_indices][:, batch_indices].to(device)
    return X_batch, A_batch

#====================================================================================
# loads a saved batch from file
#====================================================================================
def load_batch_output(
    path: str,
    map_location: Optional[Literal["cpu", "cuda"]] = "cpu"
) -> Any:
    """
    Load a saved batch output file in a memory-safe way.

    This utility wraps `torch.load` with a default `map_location` to prevent 
    loading large tensors directly to GPU by accident. It is primarily used 
    for restoring intermediate batch outputs or cached results from disk.

    Parameters
    ----------
    path : str
        File path to the saved batch output (created via `torch.save`).
    map_location : {'cpu', 'cuda', 'mps'}, default='cpu'
        Device to map tensors to when loading the file. Defaults to `'cpu'` 
        for safety and portability.

    Returns
    -------
    Any
        The deserialized Python object or tensor(s) loaded from the file.

    Examples
    --------
    >>> output = {'loss': 0.25, 'preds': torch.randn(64, 10)}
    >>> torch.save(output, "batch_out.pt")
    >>> data = load_batch_output("batch_out.pt")
    >>> data["loss"]
    0.25
    """
    return torch.load(path, map_location=map_location)

#====================================================================================
# custom function for data splitting based on graph and features + labels
#====================================================================================
def split_dataset(
    X: torch.Tensor,
    A: torch.Tensor,
    labels: List[torch.Tensor],
    split: List[float, float] = [0.8, 0.2]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Split a node-level graph dataset into training and testing sets.

    This function randomly shuffles node indices and partitions the feature matrix,
    adjacency matrix, and label tensors into training and test subsets according 
    to the provided split ratio. Sorting is applied to maintain consistent node ordering
    within each split.

    Parameters
    ----------
    X : torch.Tensor
        Node feature matrix of shape `(num_nodes, num_features)`.
    A : torch.Tensor
        Adjacency matrix of shape `(num_nodes, num_nodes)` representing graph structure.
    labels : list[torch.Tensor]
        List of label tensors for each layer or task, each of shape `(num_nodes,)`.
    split : list[float], default=[0.8, 0.2]
        Fraction of data for training and testing, respectively. Must sum to 1.0.

    Returns
    -------
    (list, list)
        - **train_set** : [train_X, train_A, labels_train]
          - `train_X` (torch.Tensor): Training node features.
          - `train_A` (torch.Tensor): Adjacency submatrix for training nodes.
          - `labels_train` (list[torch.Tensor]): Labels for training nodes.
          
        - **test_set** : [test_X, test_A, labels_test]
          - `test_X` (torch.Tensor): Test node features.
          - `test_A` (torch.Tensor): Adjacency submatrix for test nodes.
          - `labels_test` (list[torch.Tensor]): Labels for test nodes.

    Notes
    -----
    - Randomizes node order before splitting.
    - Sorting within splits ensures stable ordering.
    - Works for multi-layer or multi-task label lists.

    Examples
    --------
    >>> X = torch.randn(100, 16)
    >>> A = torch.randint(0, 2, (100, 100))
    >>> labels = [torch.randint(0, 2, (100,))]
    >>> train_set, test_set = split_dataset(X, A, labels)
    >>> [x.shape for x in train_set[:2]]
    [torch.Size([80, 16]), torch.Size([80, 80])]
    """
    # Validate input dimensions
    num_nodes = X.size(0)
    assert A.shape == (num_nodes, num_nodes), "Adjacency matrix must be square and match feature dimension"
    assert np.isclose(sum(split), 1.0, atol=1e-5), "Split ratios must sum to 1.0"

    # Determine train/test sizes
    train_size = int(np.round(split[0] * num_nodes))

    # Randomly permute node indices
    indices = torch.randperm(num_nodes)
    train_indices = torch.sort(indices[:train_size]).values
    test_indices = torch.sort(indices[train_size:]).values

    # Slice features and adjacency matrices
    train_X = X[train_indices]
    test_X = X[test_indices]
    train_A = A[train_indices][:, train_indices]
    test_A = A[test_indices][:, test_indices]

    # Slice labels for each layer
    labels_train = [lab[train_indices] for lab in labels]
    labels_test = [lab[test_indices] for lab in labels]

    # Package outputs
    train_set = [train_X, train_A, labels_train]
    test_set = [test_X, test_A, labels_test]

    return train_set, test_set



#====================================================================================
# model evaluation utility
#====================================================================================
def evaluate(
    model: Any,
    X: torch.Tensor,
    A: torch.Tensor,
    k: int,
    true_labels: Optional[List[torch.Tensor]] = None,
    run_eval: bool = True,
    device: torch.device | str = 'cpu'
) -> Tuple[
    Optional[List[torch.Tensor]], 
    Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]], 
    Optional[List[torch.Tensor]]
]:
    """
    Perform evaluation of a hierarchical graph model.

    This function evaluates the model on a graph dataset by moving tensors
    to the target device only during forward pass and immediately moving
    results back to CPU. Supports bottom-up and other hierarchical methods,
    and optionally computes performance metrics against true labels.

    Parameters
    ----------
    model : Any
        The trained graph model to evaluate. Must implement `forward(X, A)` and have `method` attribute.
    X : torch.Tensor
        Node feature matrix `(num_nodes, num_features)`.
    A : torch.Tensor
        Adjacency matrix `(num_nodes, num_nodes)`.
    k : int
        Number of layers or communities for performance evaluation.
    true_labels : list[torch.Tensor], optional
        Ground-truth labels for performance evaluation. Defaults to None.
    run_eval : bool, default=True
        If False, skips evaluation and returns None placeholders.
    device : str, default='cpu'
        Device to perform forward pass (`'cpu'`, `'cuda'`, or `'mps'`).

    Returns
    -------
    perf_layers : list[Any] | None
        Layer-wise performance metrics. None if `run_eval=False` or `true_labels` not provided.
    model_outputs : tuple
        Tuple containing CPU tensors from the model forward pass:
        `(X_pred, A_pred, X_list, A_list, P_list, S_pred, AW_pred)`.
        Entries are None if `run_eval=False`.
    S_relab : list[torch.Tensor] | None
        Relabeled predictions, ready for performance evaluation or downstream processing.
        None if `run_eval=False`.

    Notes
    -----
    - Uses `memory_efficient_context` to aggressively free memory.
    - Moves only necessary tensors to device during forward pass.
    - Supports models with `bottom_up` or other hierarchical methods.

    Example
    -------
    >>> perf, outputs, relab = evaluate_efficient(model, X, A, k=3, true_labels=labels, device="cuda")
    """
    if not run_eval:
        return None, (None, None, None, None, None, None, None), None
    
    with torch.no_grad():
        with memory_efficient_context():
            model.eval()
            # Move data to device only during forward pass
            X_device = X.to(device)
            A_device = A.to(device)
            
            output = model.forward(X_device, A_device)
            
            # Move results back to CPU immediately
            output_cpu = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    output_cpu.append(item.detach().cpu())
                elif isinstance(item, list):
                    output_cpu.append([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in item])
                else:
                    output_cpu.append(item)
            
            X_pred, A_pred, A_logit, X_list, A_list, P_list, S_pred, AW_pred = output_cpu
    
    perf_layers = []
    
    # Process predictions efficiently
    if model.method == 'bottom_up':
        S_trace_eval = trace_comms([s.clone() for s in S_pred], model.comm_sizes)
        S_all, S_temp, S_out = S_trace_eval
        S_relab = [s.detach().numpy() for s in S_temp][::-1]
    else:
        gp = [torch.unique(s, sorted=True, return_inverse=True) for s in S_pred]
        S_relab = [g[1] for g in gp]
        
    if true_labels:
        perf_layers = get_layered_performance(k, S_relab, true_labels)
        
    return perf_layers, (X_pred, A_pred, X_list, A_list, P_list, S_pred, AW_pred), S_relab


#====================================================================================
# A simple function which computes the modularity of a graph based on community assignments
#====================================================================================
def modularity(A: torch.Tensor, P: torch.Tensor, res: Optional[float] = 1.0) -> torch.Tensor:
    """
    Computes the modularity of a graph given an adjacency matrix and community assignments.

    Modularity measures the strength of division of a network into communities. 
    High modularity indicates dense connections within communities and sparse connections 
    between communities.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of the graph of shape (N, N), where N is the number of nodes.
        Should be symmetric for undirected graphs.
    P : torch.Tensor
        Community assignment matrix of shape (N, C), where C is the number of communities.
        Each row typically contains a one-hot encoding of the node's community membership.
    res : float, optional (default=1.0)
        Resolution parameter that can adjust the granularity of detected communities.
        Higher values favor smaller communities.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the modularity of the graph.

    Example
    -------
    >>> A = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    >>> P = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    >>> modularity = Modularity(A, P)
    >>> print(modularity)
    tensor(0.1667)
    """
    r = A.sum(dim = 1)
    n = A.sum()
    B = A - res*(torch.outer(r,r) / n)
    modularity = torch.trace(torch.mm(P.T, torch.mm(B, P)))/(n)
    return modularity


#====================================================================================
# within cluster loss computed using input feature matrix
#====================================================================================
def wcss(X: torch.Tensor, Plist: List[torch.Tensor], method: Literal['bottom_up', 'top_down']):
    
    """
    Computes Hierarchical Within-Cluster Sum of Squares
    X: node feature matrix N nodes by q features
    P: assignment probabilities for assigning N nodes to k clusters
    k: number of clusters
    """
    if method == 'bottom_up':
        P = torch.linalg.multi_dot(Plist)
    else:
        P = Plist
        
    N = X.shape[0]
    oneN = torch.ones(N, 1)
    M = torch.mm(torch.mm(X.T, P), torch.diag(1/torch.mm(oneN.T, P).flatten()))
    D = X.T - torch.mm(M, P.T)
    MSW = torch.sum(torch.sqrt(torch.diag(torch.mm(D.T, D))))
    return MSW, M


#====================================================================================
#This function computes the between cluster sum of squares for features X given with predicted labels 
#====================================================================================
def bcss(X: torch.Tensor, cluster_centroids: torch.Tensor, numclusts: int, norm_degree: int = 2,
         weight_by: str = ['kmeans','anova']):
    """
    X: node attribute matrix
    cluster_centroids: the centroids corresponding to a set of identified clusters
                       in X
    numclusts: number of inferred clusters in X
    norm_degree: the norm used to compute the distance
    weight_by: weighting scheme
    """
    #X_tensor = torch.tensor(X, requires_grad=True)
    supreme_centroid = torch.mean(X, dim = 0)
    pdist = nn.PairwiseDistance(p=norm_degree)
    if weight_by == 'kmeans':
        BCSS_mean_distance = torch.mean(pdist(cluster_centroids, supreme_centroid))
    else:
        BCSS_mean_distance = (1/(numclusts - 1))*torch.sum(pdist(cluster_centroids, supreme_centroid))
    
    
    return BCSS_mean_distance
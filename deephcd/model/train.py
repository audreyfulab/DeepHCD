import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optimizers 
from deephcd.utils.utilities import logging_config, trace_comms
from deephcd.utils.train_utils import get_batch_data, get_efficient_batches, evaluate, memory_efficient_context, load_batch_output, modularity, wcss
from deephcd.utils.plot_utils import plot_loss, plot_perf, plot_nodes, plot_clust_heatmaps
import os
from typing import Optional, Union, List, Literal, Dict, Any, Tuple


# ======================================================================================
# Optimized early stopping with memory management
# ======================================================================================
class EarlyStopping:
    
    """
    Implements early stopping to terminate training when validation loss fails to improve, 
    with optional checkpointing and memory-safe model saving.

    This class monitors a given loss metric during model training and stops the process once
    it ceases to improve beyond a specified tolerance (`delta`) for a number of epochs (`patience`).
    It also saves the best-performing model checkpoint safely to disk, avoiding GPU memory issues.

    Parameters
    ----------
    patience : int, default=3
        Number of consecutive epochs without improvement to tolerate before stopping training.
    verbose : bool, optional, default=False
        If True, prints progress messages when the loss improves or when patience counters increment.
    delta : int, optional, default=0
        Minimum change in the monitored loss required to qualify as an improvement.
    path : str, optional
        Directory path where the model checkpoint will be saved. Defaults to the current working directory.

    Attributes
    ----------
    counter : int
        Tracks the number of consecutive epochs without improvement.
    best_score : float | torch.Tensor | np.ndarray | None
        Stores the best loss score encountered during training.
    early_stop : bool
        Indicates whether training should be stopped early.
    loss_min : float
        Records the minimum observed loss value.
    path : str
        Directory path to save model checkpoints.

    Methods
    -------
    __call__(loss, model, _type)
        Evaluates the current loss and updates internal counters; saves checkpoint if improvement occurs.
    save_checkpoint(loss, model)
        Saves the model state to disk when a new best loss is achieved.

    Examples
    --------
    >>> stopper = EarlyStopping(patience=5, delta=0.01, verbose=True)
    >>> for epoch in range(100):
    ...     val_loss = validate(model)
    ...     stopper(val_loss, model, _type='test')
    ...     if stopper.early_stop:
    ...         print("Early stopping triggered.")
    ...         break
    """
    def __init__(self, patience: int =3, verbose: Optional[bool]=False, delta: Optional[int]=0, path: Optional[str]=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = float('inf')
        self.delta = delta
        self.path = path if path else os.getcwd()

    def __call__(self, loss: float | torch.Tensor | np.ndarray, model: nn.Module, _type: Optional[Literal['test', 'total']]):
        """Evaluates the current loss and decide whether to continue training.

        Parameters
        ----------
        loss : float | torch.Tensor | np.ndarray
            Current loss value to monitor for improvement.
        model : nn.Module
            PyTorch model being trained. A checkpoint is saved if the loss improves.
        _type : {'test', 'total'}, optional
            Label indicating the type of loss being monitored; used only for display/logging.

        Returns
        -------
        None
        """
        score = loss
        self._type = _type
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss: float | torch.Tensor | np.ndarray, model: nn.Module):
        """Save the model checkpoint when validation loss improves.

        Parameters
        ----------
        loss : float | torch.Tensor | np.ndarray
            Current loss value.
        model : nn.Module
            PyTorch model to be checkpointed.

        Returns
        -------
        None
        """
        if self.verbose:
            print(f'\n{self._type} loss decreased ({self.loss_min:.6f} --> {loss:.6f}). Saving model...\n')
        # Save to CPU to avoid GPU memory issues
        torch.save(model.cpu().state_dict(), os.path.join(self.path, 'checkpoint.pth'))
        model.to(next(model.parameters()).device)  # Move back to original device
        self.loss_min = loss


# ======================================================================================
# Optimized HCD output class with lazy loading
# ======================================================================================
class HCD_output:
    """
    Memory-optimized container for storing and inspecting Hierarchical Community Detection (HCD) 
    model outputs, intermediate representations, and training histories.

    This class encapsulates model outputs, reconstructed features, attention weights, and 
    performance histories from hierarchical clustering models. It performs safe detachment 
    and CPU transfer of tensors to minimize GPU memory usage, and supports lazy reloading 
    of historical model outputs.

    Parameters
    ----------
    X : torch.Tensor
        Input feature matrix used during training.
    A : torch.Tensor
        Input adjacency matrix representing graph structure.
    test_set : tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        Optional test set containing `(X_test, A_test, labels_test)`.
    labels : torch.Tensor | np.ndarray
        Labels for the training samples.
    model_output : tuple
        Model output tuple containing:
        `(X_final, A_final, _, X_all_final, A_all_final, P_all_final, S_final, AW_final)`.
    test_history : list[float]
        Recorded loss values from test epochs.
    train_history : list[float]
        Recorded loss values from training epochs.
    perf_history : list[Any]
        Performance metrics tracked over epochs.
    pred_history : list[Any]
        Prediction history over epochs.
    batch_indices : list[torch.Tensor] | None
        Batch index tensors used during training.
    device : str, default='cpu'
        Target device for storing final tensors.

    Attributes
    ----------
    reconstructed_features : torch.Tensor
        Final reconstructed node features from the model.
    reconstructed_adj : torch.Tensor
        Final reconstructed adjacency matrix.
    latent_features : torch.Tensor
        Latent representations of the top-level graph.
    partitioned_data : list[torch.Tensor]
        Partitioned feature tensors for subgraphs.
    attention_weights : dict[str, list[list[torch.Tensor]]] | None
        Nested attention weights detached from GPU memory.
    train_loss_history, test_loss_history, performance_history, pred_history : list
        Historical metrics from model training.
    probabilities : dict[str, Any]
        Hierarchical prediction probabilities at top and intermediate levels.
    adjacency : dict[str, list[torch.Tensor]]
        Hierarchical adjacency matrices for community and partitioned graphs.
    predicted_train : dict[str, torch.Tensor]
        Model predictions for training data at multiple hierarchy levels.
    batch_indices : list[torch.Tensor] | None
        Batch indices mapped to CPU.
    model_output_history : list[tuple]
        Record of key model outputs for lazy inspection.
    best_loss_index : int | None
        Index of best epoch by loss (set externally).
    hierarchical_clustering_preds, louvain_preds, kmeans_preds : Any
        Placeholder attributes for clustering outputs.
    table, perf_table : Any
        Optional tabular summaries of results.

    Methods
    -------
    to_dict()
        Return all attributes as a dictionary.
    load_history_item(idx, map_location='cpu')
        Retrieve or lazily load a stored model output from history.
    show_results()
        Display the performance summary table if available.
    """

    
    def __init__(self,
        X: torch.Tensor,
        A: torch.Tensor,
        test_set: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        labels: Union[torch.Tensor, np.ndarray],
        model_output: tuple[
            torch.Tensor, torch.Tensor, Any, tuple, tuple, tuple, tuple, Optional[dict[str, list[list[torch.Tensor]]]]
        ],
        test_history: list[torch.Tensor | np.ndarray],
        train_history: list[torch.Tensor | np.ndarray],
        perf_history: list[torch.Tensor | np.ndarray],
        pred_history: list[torch.Tensor | np.ndarray],
        batch_indices: Optional[list[torch.Tensor]],
        ):
        
        # Store only essential data, move to CPU immediately
        X_final, A_final, _, X_all_final, A_all_final, P_all_final, S_final, AW_final = model_output
        eval_X, eval_A, eval_labels = test_set if test_set else (None, None, None)

        self.model_output_history = [
            (X_final, A_final, X_all_final, A_all_final, P_all_final, S_final, AW_final)
        ]

        # Move tensors to CPU and detach to free GPU memory
        self.attention_weights = {
            k: [[t.detach().cpu() for t in head_list] for head_list in v]
            for k, v in AW_final.items()
        } if AW_final is not None else None


        self.reconstructed_features = X_final.detach().cpu()
        self.reconstructed_adj = A_final.detach().cpu()
        self.latent_features = X_all_final[0].detach().cpu()
        
        # Handle partitioned data efficiently
        self.partitioned_data = [x.detach().cpu() for x in X_all_final[1]]
        self.partitioned_latent_features = [x.detach().cpu() for x in X_all_final[2]]
        
        # Store histories (these are small)
        self.train_loss_history = train_history
        self.test_loss_history = test_history
        self.performance_history = perf_history
        self.pred_history = pred_history
        
        # Store data references
        self.training_data = {
            'X_train': X.detach().cpu(), 
            'A_train': A.detach().cpu(), 
            'labels_train': labels
        }
        
        if test_set:
            self.test_data = {
                'X_test': eval_X.detach().cpu() if eval_X is not None else None,
                'A_test': eval_A.detach().cpu() if eval_A is not None else None,
                'labels_test': eval_labels
            }
        else:
            self.test_data = {'X_test': None, 'A_test': None, 'labels_test': None}
            
        self.probabilities = {
            'top': P_all_final[0].detach().cpu(),
            'middle': [p.detach().cpu() for p in P_all_final[1]]
        }
        
        self.adjacency = {
            'community_graphs': [a.detach().cpu() for a in A_all_final[1]],
            'partitioned_graphs': [a.detach().cpu() for a in A_all_final[2]]
        }
        
        self.predicted_train = {
            'top': S_final[0].detach().cpu(),
            'middle': S_final[1].detach().cpu()
        }
        
        self.batch_indices = [idx.detach().cpu() for idx in batch_indices] if batch_indices else None
        
        # Initialize additional attributes
        self.best_loss_index = None
        self.hierarchical_clustering_preds = None
        self.louvain_preds = None
        self.kmeans_preds = None
        self.table = None
        self.perf_table = None

    # ------------------------------
    # Utility methods
    # ------------------------------
    
    def to_dict(self) -> dict[str, Any]:
        """Return all attributes of the object as a dictionary."""
        return {attr: getattr(self, attr) for attr in self.__dict__}
    
    def load_history_item(self, idx: int, map_location: Optional[str] ="cpu") -> List[torch.Tensor | np.ndarray]:
        """Load a stored or lazily retrievable model output from history."""
        idx = min(idx, len(self.model_output_history) - 1)
        ref = self.model_output_history[idx]
        if isinstance(ref, dict) and "file" in ref:
            return load_batch_output(ref["file"], map_location=map_location)
        return ref
    def show_results(self) -> None:
        """Print the performance summary table if available."""
        if self.perf_table is not None:
            print(self.perf_table)


# ======================================================================================
# Modularity loss functions
# ======================================================================================
class OptimizedModularityLoss(nn.Module):
    def __init__(self):
        super(OptimizedModularityLoss, self).__init__()
        
    def forward(self, all_A, all_P, resolutions=None):
        loss = 0.0
        loss_list = []
        
        for index, (A, P) in enumerate(zip(all_A, all_P)):
            resolution = resolutions[index] if resolutions else 1.0
            
            with memory_efficient_context():
                mod = Modularity(A, P, resolution)
                loss += mod
                loss_list.append(float(mod.detach().cpu().numpy()))
                
        return loss, loss_list


# ======================================================================================
# cluster based loss function
# ======================================================================================
class OptimizedClusterLoss(nn.Module):
    def __init__(self):
        super(OptimizedClusterLoss, self).__init__()

    def forward(self, Lamb, Attributes, Probabilities, method):
        loss = 0.0
        loss_list = []
        
        if not isinstance(Attributes, list):
            N = Attributes.shape[0]
            ptensor_list = [torch.eye(N, device=Attributes.device)]
            
        for idx, P in enumerate(Probabilities):
            Attr = Attributes[idx] if isinstance(Attributes, list) else Attributes
            
            if method == 'bottom_up':
                ptensor_list.append(P)
            else:
                ptensor_list = P
                
            with memory_efficient_context():
                within_ss, centroids = WCSS(X=Attr, Plist=ptensor_list, method=method)
                
                weight = Lamb[idx] if isinstance(Lamb, list) else Lamb
                weighted_loss = weight * within_ss
                
                loss_list.append(float(weighted_loss.detach().cpu().numpy()))
                loss += weighted_loss

        return loss, loss_list


# training function

class Trainer():
    """
    Trains the HRGNgene model on the given dataset.

    This is a trainer class for the DeepHCD  model using modularity-based and clustering-based loss terms 
    while performing hierarchical clustering on gene regulatory networks (GRNs). It supports batch learning, 
    early stopping, and modularity-based clustering loss optimization.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to be trained.
    X : array-like (torch.Tensor)
        Feature matrix of shape (N, F), where N is the number of samples and F is the number of features.
    A : array-like (torch.Tensor)
        Adjacency matrix of the input graph, representing connections between samples.
    optimizer : str, optional (default='Adam')
        The optimization algorithm to use for training (e.g., 'Adam', 'SGD').
    epochs : int, optional (default=100)
        Number of training epochs.
    update_interval : int, optional (default=10)
        Frequency (in epochs) at which performance metrics are updated and logged.
    lr : float, optional (default=1e-4)
        Learning rate for the optimizer.
    gamma : float, optional (default=1)
        Weighting factor for the attribute reconstruction loss term.
    delta : float, optional (default=1)
        Weighting factor for the modularity loss term.
    lamb : float, optional (default=1)
        Weighting factor for the clustering loss term.
    layer_resolutions : list of float, optional (default=[1,1])
        Resolution parameters for modularity calculation at different hierarchical layers.
    k : int, optional (default=2)
        Number of hierarchical clustering levels.
    use_batch_learning : bool, optional (default=True)
        Whether to use mini-batch training or full-batch training.
    batch_size : int, optional (default=64)
        Size of each batch if `use_batch_learning` is enabled.
    early_stopping : bool, optional (default=False)
        Whether to stop training early if no improvement is observed.
    patience : int, optional (default=5)
        Number of epochs to wait for improvement before stopping (if early stopping is enabled).
    true_labels : array-like, optional (default=None)
        Ground truth labels used for evaluation.
    turn_off_A_loss : bool, optional (default=False)
        Whether to disable the loss term related to adjacency matrix reconstruction.
    validation_data : tuple, optional (default=None)
        Validation dataset provided as (X_val, A_val, val_labels).
    test_data : tuple, optional (default=None)
        Test dataset provided as (X_test, A_test, test_labels).
    save_output : bool, optional (default=False)
        Whether to save the model's outputs and training history.
    output_path : str, optional (default='')
        Directory path for saving model output, if enabled.
        Node size for graph visualizations.
    verbose : bool, optional (default=True)
        Whether to print detailed logs and progress updates during training.
    **kwargs : dict
        Additional keyword arguments for customizing training behavior.

    Returns:
    --------
    output : HCD_output
        An instance of `HCD_output`, which contains:
        - `all_model_output`: List of all model outputs over training.
        - `attention_weights`: Attention weights from the final model.
        - `train_loss_history`: History of training losses.
        - `test_loss_history`: History of test losses.
        - `performance_history`: Performance metrics over epochs.
        - `latent_features`: Extracted latent feature representations.
        - `partitioned_data`: Data after partitioning into hierarchical clusters.
        - `partitioned_latent_features`: Partitioned latent features at different levels.
        - `training_data`: Dictionary containing training feature matrix and adjacency matrix.
        - `test_data`: Dictionary containing test feature matrix and adjacency matrix.
        - `probabilities`: Cluster membership probabilities at different hierarchy levels.
        - `pred_history`: Predicted cluster assignments over epochs.
        - `adjacency`: Graph adjacency structures at different clustering levels.



    Notes:
    ------
    - Supports early stopping based on total loss or test loss.
    - Supports unsupervised learning based on total loss supervised learning based on validation loss when labels are provided.
    - Batch learning is enabled by default but can be disabled for full-batch training.

    Example:
    --------
    >>> from model.model import HCD
    >>> from model.train import fit
    >>> model = HCD()  # Initialize your model
    >>> X = torch.rand(100, 20)  # Example feature matrix
    >>> A = torch.randint(0, 2, (100, 100))  # Example adjacency matrix
    >>> trainer = Trainer(model, X, A, epochs=50, lr=1e-3, batch_size=32, early_stopping=True, patience=5)

    """
    
    def __init__(self, 
                model: nn.Module, 
                X: torch.Tensor, 
                A: torch.Tensor, 
                optimizer: Optional[str]='Adam', 
                optimizer_weight_decay: Optional[float]=5e-4,
                epochs: Optional[int]=50, 
                update_interval: Optional[int]=10, 
                learning_rate: Optional[float]=1e-4, 
                gamma: Optional[int]=1, 
                delta: Optional[int]=1, 
                _lambda: Optional[int]=1, 
                graph_resolutions: Optional[List[int,int,]]=[1,1], 
                k: Optional[int]=2,  
                batch_size: Optional[int]=64, 
                early_stopping: Optional[bool]=False, 
                patience: Optional[int]=5, 
                use_batch_learning: Optional[bool]=True,
                true_labels: List[str | int] | np.ndarray | torch.Tensor=None, 
                validation_data: Optional[Dict[str, torch.Tensor]]=None, 
                test_data: Optional[Dict[str, torch.Tensor]]=None, 
                save_output: Optional[bool]=False, 
                output_path='./save/path/to/output', 
                verbose: Optional[bool]=True, 
                ):
        """
        Initializes the Trainer class with optional configurations.

        Notes:
        - X and A are **not moved to device** until training/evaluation to save memory.
        - Histories (loss, predictions, performance) are stored on CPU only.
        """
        # Store model and move to device only during training/evaluation
        self.model = model

        # Store dataset references (keep on CPU, move to device only in training/eval)
        self.X = X
        self.A = A
        self.true_labels = true_labels
        self.validation_data = validation_data
        self.test_data = test_data

        # Training hyperparameters
        self.optimizer_type = optimizer
        self.optimizer_weight_decay = optimizer_weight_decay
        self.epochs = epochs
        self.update_interval = update_interval
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.delta = delta
        self._lambda = _lambda
        self.graph_resolutions = graph_resolutions
        self.k = k
        self.batch_size = batch_size
        self.use_batch_learning = use_batch_learning
        self.early_stopping = early_stopping
        self.patience = patience
        self.save_output = save_output
        self.output_path = output_path
        self.verbose = verbose

        # Initialize memory-efficient histories (stored on CPU only)
        self.train_loss_history: List[Dict[str, float]] = []
        self.test_loss_history: List[Dict[str, float]] = []
        self.performance_history: List[Optional[List]] = []
        self.pred_history: List[Optional[List]] = []

        # Placeholder for batch indices (generated once, CPU)
        self.batch_indices_list: Optional[List[torch.Tensor | np.ndarray]] = None
        
    def fit(self, device: torch.device):
        """
        Optimized training function with memory efficiency improvements
        """
        
        # Move model to device
        model = model.to(device)
        
        # Initialize storage with minimal memory footprint
        train_loss_history = []
        perf_hist = []
        pred_list = []
        test_loss_history = []
        
        comm_layers = len(model.comm_sizes)
        
        # Early stopping
        if self.early_stopping:
            early_stop = EarlyStopping(patience=self.patience, 
                                       verbose=True, 
                                       path=self.output_path)
        
        # Optimizer
        optimizer = optimizers.Adam(model.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.optimizer_weight_decay)
        
        # Loss functions
        A_recon_loss = nn.BCELoss(reduction='mean')
        X_recon_loss = nn.MSELoss(reduction='mean')
        modularity_loss_fn = OptimizedModularityLoss()
        clustering_loss_fn = OptimizedClusterLoss()
        
        # Generate batch indices once (memory efficient)
        if self.use_batch_learning:
            if self.batch_size > self.X.shape[0]:
                raise ValueError(f'Batch size ({self.batch_size}) larger than dataset size ({self.X.shape[0]})')
            self.batch_indices_list = get_efficient_batches(X, A, self.batch_size, device='cpu')
        else:
            self.batch_indices_list = [torch.arange(self.X.shape[0])]
        
        print(f"Training on {len(batch_indices_list)} batches")
        
        # Training loop
        for epoch in range(self.epochs):
            model.train()
            epoch_start = time.time()
            
            # Initialize epoch losses
            total_loss = 0.0
            train_epoch_losses = {
                'A': 0.0, 'X': 0.0, 
                'clust': [0.0] * len(model.comm_sizes), 
                'mod': [0.0] * len(model.comm_sizes)
            }
            
            print(f'Epoch {epoch + 1}/{epochs}')
            print('=' * 50)
            
            # Batch processing with memory management
            for batch_idx, batch_indices in enumerate(batch_indices_list):
                with memory_efficient_context():
                    # Get batch data on device
                    A = A / A.max()
                    X_batch, A_batch = get_batch_data(X, A, batch_indices, device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    forward_output = model.forward(X_batch, A_batch)
                    X_hat, A_hat, A_logit, X_all, A_all, P_all, S_all, AW = forward_output
                    
                    # Compute losses efficiently
                    mod_clust_output = get_optimized_losses(
                        model, X_batch, A_batch, forward_output, lamb, 
                        layer_resolutions, modularity_loss_fn, clustering_loss_fn
                    )
                    Mod_loss, Modloss_values, Clust_loss, Clustloss_values = mod_clust_output
                    
                    
                    # Reconstruction losses
                    A_hat = torch.clamp(A_hat, min=1e-7, max=1 - 1e-7)
                    X_loss = X_recon_loss(X_hat, X_batch)
                    A_loss = A_recon_loss(A_hat, A_batch)
                    
                    # Total loss
                    batch_loss = A_loss + gamma * X_loss + Clust_loss - delta * Mod_loss
                    print(f'A_loss: ',A_loss,', X_loss: ',X_loss,', Clust_loss',Clust_loss,', Mod_loss: ',Mod_loss)
                    # Backward pass
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Update epoch losses
                    total_loss += batch_loss.item()
                    print(f'batch loss: ',batch_loss.item())
                    train_epoch_losses['A'] += A_loss.item()
                    train_epoch_losses['X'] += X_loss.item()
                    
                    for i, (c, m) in enumerate(zip(Clustloss_values, Modloss_values)):
                        if i < len(train_epoch_losses['clust']):
                            train_epoch_losses['clust'][i] += c
                            train_epoch_losses['mod'][i] += m
                    
                    # Clear batch data from GPU
                    del X_batch, A_batch, forward_output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Store training history
            train_loss_history.append({
                'Total Loss': total_loss,
                'A Reconstruction': train_epoch_losses['A'],
                'X Reconstruction': gamma * train_epoch_losses['X'],
                'Modularity': delta * np.array(train_epoch_losses['mod']),
                'Clustering': np.array(train_epoch_losses['clust'])
            })
            
            # Evaluation (less frequent to save memory)
            test_loss = 0.0
            if test_data:
                eval_X, eval_A, eval_labels = test_data
                with memory_efficient_context():
                    test_perf, test_output, S_replab_test = evaluate_efficient(
                        model, eval_X, eval_A, k, eval_labels, device=device
                    )
                    
                    if test_output[0] is not None:
                        X_hat_test, A_hat_test = test_output[0], test_output[1]
                        eval_X_dev = eval_X.to(device)
                        eval_A_dev = eval_A.to(device)
                        X_hat_dev = X_hat_test.to(device)
                        A_hat_dev = A_hat_test.to(device)
                        
                        X_loss_test = X_recon_loss(X_hat_dev, eval_X_dev).item()
                        A_loss_test = A_recon_loss(A_hat_dev, eval_A_dev).item()
                        print(A_hat)
                        test_loss = A_loss_test + gamma * X_loss_test
                        
            
            test_loss_history.append({'Total Loss': test_loss})
            
            # Performance evaluation (periodic)
            if epoch % update_interval == 0:
                with memory_efficient_context():
                    train_perf, eval_output, S_eval = evaluate_efficient(
                        model, X, A, k, true_labels, device=device
                    )
                    perf_hist.append(train_perf)
                    pred_list.append(S_eval)
                    
                    if true_labels:
                        print('\nMODEL PERFORMANCE')
                        print_performance_efficient(perf_hist, comm_layers, k)
            
            # Early stopping check
            if early_stopping:
                print('Early Stopping Start\n')
                print(f'A_loss_test: ',A_loss_test, ', X_loss_test: ', X_loss_test)
                print(f'Total Loss: ',  total_loss)
                print(f'Test Loss: ', test_loss)
            
                early_stop(test_loss if test_data else total_loss, model)
                if early_stop.early_stop:
                    print("Early stopping triggered")
                    break
            
            epoch_time = time.time() - epoch_start
            if verbose:
                print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s')
                print(f'Total Loss: {total_loss:.4f}')
                print('-' * 50)
        
        # Final model output
        print("Generating final output...")
        with memory_efficient_context():
            model.eval()
            with torch.no_grad():
                final_out = model.forward(X.to(device), A.to(device))
                # Move to CPU immediately
                final_out_cpu = []
                for item in final_out:
                    if isinstance(item, torch.Tensor):
                        final_out_cpu.append(item.detach().cpu())
                    elif isinstance(item, list):
                        final_out_cpu.append([x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in item])
                    else:
                        final_out_cpu.append(item)
        
        # Create optimized output object
        output = HCD_output(
            X=X, A=A, test_set=test_data, labels=true_labels,
            model_output=final_out_cpu, train_history=train_loss_history,
            test_history=test_loss_history, perf_history=perf_hist,
            pred_history=pred_list, batch_indices=batch_indices_list, device='cpu'
        )
        
        return output

    def get_optimized_losses(model, Xbatch, Abatch, output, lamb, resolution, modlossfn, clustlossfn):
        """Optimized loss computation with memory management"""
        X_hat, A_hat, A_logit, X_all, A_all, P_all, S_all, AW = output
        
        if model.method == 'bottom_up':
            S_sub, S_relab, S = trace_comms([s.clone() for s in S_all], model.comm_sizes)
            Mod_loss, Modloss_values = modlossfn([Abatch] + A_all[1], P_all, resolution)
            Clust_loss, Clustloss_values = clustlossfn(lamb, Xbatch, P_all, model.method)
        elif model.method == "top_down":
            # Top-down processing
            top_mod_loss, values_top = modlossfn([A_all[0]], [P_all[0]], resolution)
            middle_mod_loss, values_mid = modlossfn(A_all[-1], P_all[1], resolution)
            Mod_loss = top_mod_loss + middle_mod_loss
            Modloss_values = values_top + [torch.mean(torch.tensor(values_mid)).item()]
            
            Clust_loss_top, Clustloss_values_top = clustlossfn(lamb[0], Xbatch, [P_all[0]], model.method)
            Clust_loss_mid, Clustloss_values_mid = clustlossfn(lamb[1], X_all[-1], P_all[1], model.method)
            Clust_loss = Clust_loss_top + Clust_loss_mid
            Clustloss_values = Clustloss_values_top + [torch.sum(torch.tensor(Clustloss_values_mid)).item()]
        
        return Mod_loss, Modloss_values, Clust_loss, Clustloss_values

    def print_performance_efficient(history, comm_layers, k):
        """Efficient performance printing with error handling"""
        if not history or all(h is None for h in history):
            print("No performance history available")
            return

        valid_history = [h for h in history if h is not None]
        if not valid_history:
            print("No valid performance data available")
            return

        last_perf = valid_history[-1]
        layer_names = ['top'] + [f'middle_{i}' for i in range(comm_layers-1)]
        
        for i in range(min(k, len(last_perf))):
            if i >= len(last_perf) or last_perf[i] is None:
                print(f"No data available for {layer_names[i]} layer")
                continue
                
            print(f'{"-"*20} {layer_names[i]} layer {"-"*20}')
            
            metrics = last_perf[i]
            metric_names = ['Homogeneity', 'Completeness', 'NMI', 'ARI']
            
            for j, (name, value) in enumerate(zip(metric_names, metrics[:4])):
                print(f'{name}: {value:.4f}')
            print('-' * 50)
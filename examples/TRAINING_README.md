# DeepHCD Training and Inference Guide

This guide covers how to train DeepHCD models for hierarchical community detection and use them for inference.

## Quick Start

### Option 1: Minimal Training Example

Train a model on simulated data with default settings:

```bash
python simple_training.py
```

Requires: `./very_small_graph_150/` directory with simulated data

### Option 2: Full Training with Options

Train with customizable parameters:

```bash
python train_hierarchy_example.py \
    --data_path ./very_small_graph_150/ \
    --epochs 50 \
    --output_path ./my_training/
```

### Option 3: Inference on Trained Model

Run predictions using a saved model:

```bash
python inference_example.py \
    --model_path ./my_training/trained_model.pth \
    --data_path ./test_data/
```

## Training Workflow

### 1. Data Preparation

DeepHCD expects:
- **Feature matrix (X)**: Nodes × Features (e.g., gene expression data)
- **Adjacency matrix (A)**: Nodes × Nodes (network structure)
- **Optional labels**: For evaluation during training

```python
from deephcd.utils.utilities import LoadData

# Load from simulated data
pe, adj, idx_top, idx_mid, labels, sorted_top, sorted_mid = LoadData('./data_path/')

# Prepare inputs
X = torch.FloatTensor(pe[idx_mid, :])  # Features
A = torch.FloatTensor(adj[2]) + torch.eye(X.shape[0])  # Adjacency with self-loops
```

### 2. Estimate Optimal Communities

Use automatic community size estimation:

```python
from deephcd.utils.utilities import compute_kappa

comm_sizes = compute_kappa(
    X, A,
    method='bethe_hessian',  # or 'elbow', 'silouette'
    verbose=True
)
print(f"Estimated communities: {comm_sizes}")
```

Or specify manually:
```python
comm_sizes = [15, 5]  # [middle layer, top layer]
```

### 3. Create Model

```python
from deephcd.model.model import HCD

model = HCD(
    nodes=X.shape[0],
    attrib=X.shape[1],
    method='top_down',              # or 'bottom_up'
    ae_hidden_dims=[256, 128],      # Autoencoder layers
    comm_sizes=comm_sizes,          # Community sizes
    ae_operator='GATv2Conv',        # Graph attention operator
    ae_attn_heads=5,                # Attention heads
    dropout=0.2,
    normalize_input=True
).to(device)
```

### 4. Train Model

```python
from deephcd.model.train import Trainer

trainer = Trainer(
    model=model,
    X=X,
    A=A,
    epochs=50,
    learning_rate=1e-3,
    batch_size=32,
    early_stopping=True,
    patience=5,
    true_labels=[sorted_top, sorted_mid],  # Optional, for evaluation
    verbose=True
)

output = trainer.fit(device)
```

### 5. Access Results

```python
# Predicted community assignments
top_communities = output.predicted_train['top']
middle_communities = output.predicted_train['middle']

# Assignment probabilities
top_probs = output.probabilities['top']
middle_probs = output.probabilities['middle']

# Training history
losses = output.train_loss_history
performance = output.performance_history

# Reconstructed data
X_reconstructed = output.reconstructed_features
A_reconstructed = output.reconstructed_adj
```

## Model Architecture Options

### Graph Neural Network Operators

Choose from different GNN architectures:

- **GATv2Conv** (default): Graph Attention Networks v2 - best for most cases
- **GATConv**: Original Graph Attention Networks
- **SAGEConv**: GraphSAGE - good for large graphs

### Hierarchical Methods

- **top_down** (default): Start from full graph, recursively partition
- **bottom_up**: Start from individual nodes, merge into communities

### Architecture Parameters

```python
model = HCD(
    nodes=150,                      # Number of nodes
    attrib=500,                     # Feature dimensions
    method='top_down',
    ae_hidden_dims=[256, 128, 64],  # Hidden layer sizes (flexible)
    comm_sizes=[15, 5],             # Communities [middle, top]
    ae_operator='GATv2Conv',
    ae_attn_heads=5,                # More heads = more expressiveness
    dropout=0.2,                    # Regularization
    normalize_input=True,           # Normalize features
    normalize_outputs=True,         # Normalize layer outputs
    heads=1                         # Output heads
)
```

## Training Parameters

### Basic Training

```python
trainer = Trainer(
    model=model,
    X=X,                            # Features
    A=A,                            # Adjacency
    epochs=50,                      # Training iterations
    learning_rate=1e-3,             # Learning rate
    batch_size=32,                  # Mini-batch size
    use_batch_learning=True         # Enable batching
)
```

### Loss Hyperparameters

Control the balance between different objectives:

```python
trainer = Trainer(
    ...,
    gamma=2.0,                      # Feature reconstruction weight
    delta=10.0,                     # Modularity loss weight
    _lambda=[1/60, 1/20]           # Clustering loss [middle, top]
)
```

- **gamma**: Higher = better feature reconstruction
- **delta**: Higher = more emphasis on network structure
- **lambda**: Higher = tighter, more distinct communities

### Early Stopping

Prevent overfitting:

```python
trainer = Trainer(
    ...,
    early_stopping=True,
    patience=5                      # Stop if no improvement for 5 epochs
)
```

### Train/Test Splitting

Evaluate generalization:

```python
from deephcd.utils.train_utils import split_dataset

train, test = split_dataset(X, A, labels, split=[0.8, 0.2])
X_train, A_train, labels_train = train

trainer = Trainer(
    model=model,
    X=X_train,
    A=A_train,
    true_labels=labels_train,
    test_data=test                  # Evaluate on test set each epoch
)
```

## Inference and Prediction

### Load Trained Model

```python
checkpoint = torch.load('trained_model.pth')

# Recreate model
model = HCD(...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Run Predictions

```python
with torch.no_grad():
    X_hat, A_hat, _, X_all, A_all, P_all, S_all, _ = model.forward(X, A)

    # Community predictions
    top_predictions = S_all[0]
    middle_predictions = S_all[1]

    # Probability distributions
    top_probabilities = P_all[0]
    middle_probabilities = P_all[1]
```

### Interpret Results

```python
# Number of detected communities
n_top = len(torch.unique(top_predictions))
n_middle = len(torch.unique(middle_predictions))

# Confidence scores (average max probability)
confidence = top_probabilities.max(dim=1)[0].mean()

# Community sizes
for comm in torch.unique(top_predictions):
    size = (top_predictions == comm).sum()
    print(f"Community {comm}: {size} nodes")
```

## Evaluation Metrics

When true labels are available:

```python
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score
)

nmi = normalized_mutual_info_score(true_labels, predictions)
ari = adjusted_rand_score(true_labels, predictions)
homogeneity = homogeneity_score(true_labels, predictions)
completeness = completeness_score(true_labels, predictions)
```

- **NMI (Normalized Mutual Information)**: 0-1, measures agreement
- **ARI (Adjusted Rand Index)**: -1 to 1, measures similarity (1=perfect)
- **Homogeneity**: Communities contain only members of single true class
- **Completeness**: All members of true class assigned to same community

## Common Use Cases

### 1. Gene Co-expression Networks

```python
# Load expression data
X = gene_expression  # Genes × Samples
A = correlation_matrix + torch.eye(n_genes)

# Train with appropriate parameters
model = HCD(
    nodes=n_genes,
    attrib=n_samples,
    ae_hidden_dims=[512, 256, 128],  # Larger for many features
    comm_sizes=[50, 10],              # Many genes, fewer modules
    dropout=0.3                        # Higher dropout for regularization
)
```

### 2. Social Networks

```python
# Use network structure directly
A = adjacency_matrix + torch.eye(n_users)
X = user_features  # Demographics, activity patterns, etc.

model = HCD(
    method='top_down',
    ae_hidden_dims=[128, 64],         # Smaller for simpler features
    ae_operator='SAGEConv'            # Good for social graphs
)
```

### 3. Citation Networks

```python
# Papers as nodes, citations as edges
X = paper_features  # Keywords, abstract embeddings
A = citation_matrix + torch.eye(n_papers)

model = HCD(
    ae_operator='GATv2Conv',          # Attention useful for citations
    ae_attn_heads=8,                   # Multiple attention patterns
    comm_sizes=[100, 20]               # Research areas → broader fields
)
```

## Troubleshooting

### Issue: Model not converging

**Solutions:**
- Reduce learning rate: `learning_rate=1e-4`
- Increase batch size: `batch_size=64`
- Adjust loss weights: reduce `gamma`, `delta`, or `lambda`
- Add more regularization: increase `dropout`

### Issue: Poor clustering performance

**Solutions:**
- Check community size estimates: try `compute_kappa()` with different methods
- Adjust `delta` (modularity weight): higher = more structure-driven
- Try different GNN operator: GATv2Conv usually works best
- Increase model capacity: larger `ae_hidden_dims`

### Issue: Overfitting

**Solutions:**
- Enable early stopping: `early_stopping=True, patience=5`
- Increase dropout: `dropout=0.3` or higher
- Reduce model size: smaller `ae_hidden_dims`
- Add train/test split to monitor generalization

### Issue: Out of memory

**Solutions:**
- Reduce batch size: `batch_size=16`
- Use smaller hidden dimensions
- Use CPU: `device='cpu'`
- Process on smaller subgraphs if possible

## Best Practices

1. **Always normalize input features**: `normalize_input=True`
2. **Start with automatic community estimation**: `compute_kappa()`
3. **Use early stopping**: Prevents overfitting
4. **Monitor multiple metrics**: Don't rely on loss alone
5. **Save your models**: `torch.save()` after successful training
6. **Document hyperparameters**: Save config with model
7. **Validate on held-out data**: Use train/test split
8. **Try multiple seeds**: Results can vary with initialization

## Example Workflow

Complete example from data to inference:

```bash
# 1. Simulate data
python simple_simulation.py

# 2. Train model
python train_hierarchy_example.py \
    --data_path ./my_simulated_network/ \
    --epochs 50 \
    --batch_size 32 \
    --early_stopping \
    --save_model \
    --output_path ./training_output/

# 3. Run inference
python inference_example.py \
    --model_path ./training_output/trained_model.pth \
    --data_path ./test_data/
```

## Advanced Topics

### Custom Loss Weighting

Fine-tune loss components:

```python
trainer = Trainer(
    ...,
    gamma=1.0,          # Feature reconstruction (lower = less emphasis)
    delta=20.0,         # Modularity (higher = more structure-based)
    _lambda=[1/30, 1/10]  # Clustering tightness [middle, top]
)
```

### Multi-GPU Training

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
trainer.fit('cuda')
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

# After creating trainer
scheduler = ReduceLROnPlateau(trainer.optimizer, 'min', patience=3)

# In training loop
for epoch in range(epochs):
    loss = trainer.train_epoch()
    scheduler.step(loss)
```

## References

- See `main.py` for a complete end-to-end example
- See `simple_simulation.py` for data generation
- See model architecture: `deephcd/model/model.py`
- See training implementation: `deephcd/model/train.py`

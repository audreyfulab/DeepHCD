#!/usr/bin/env python3
"""
Minimal Example: Train HCD Model on Simulated Data

This example shows the minimal code needed to:
1. Load simulated hierarchical network data
2. Train a DeepHCD model
3. View the results

Perfect for getting started quickly!
"""

import os
import torch
from deephcd.model.model import HCD
from deephcd.model.train import Trainer
from deephcd.utils.utilities import LoadData, compute_kappa
from deephcd.utils.train_utils import split_dataset
import time

start_time = time.perf_counter()
# Configuration
DATA_PATH = '/Users/jordandavis/Documents/DeepHCD_copy/DeepHCD/examples/very_small_graph_150'  # Path to simulated data
#DATA_PATH = '/Users/audreyq.fu/Documents/GRN/Data/1k_node_graph/'
OUTPUT_PATH = './test_nonzero_loss/'      # Path for training outputs
#OUTPUT_PATH = '/Users/audreyq.fu/Documents/GRN/Data/1k_node_graph/training_output/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*70)
print("TRAINING DEEPHCD MODEL - MINIMAL EXAMPLE")
print("="*70)

# 1. Load data
print("\n1. Loading data...")
pe, adj, idx_top, idx_mid, labels, sorted_top, sorted_mid = LoadData(DATA_PATH)

# Prepare inputs
X = torch.FloatTensor(pe[idx_mid, :])  # Features (sorted by hierarchy)
A = torch.FloatTensor(adj[:X.shape[0], :X.shape[0]]) + torch.eye(X.shape[0])  # Adjacency
nodes, features = X.shape

print(f"   Data loaded: {nodes} nodes, {features} features")

# 2. Estimate optimal number of communities
print("\n2. Estimating optimal community sizes...")
comm_sizes = compute_kappa(X, A, method='bethe_hessian', verbose=False)
print(f"   Estimated communities: {comm_sizes}")

# 3. Create model

#making labels a list of tensors
labels_list = []
if len(sorted_top) > 0:
    labels_list.append(torch.LongTensor(sorted_top))
if len(sorted_mid) > 0:
    labels_list.append(torch.LongTensor(sorted_mid))
labels_list = labels_list if labels_list else None

train_set, val_set = split_dataset(X,A,labels_list)

print("\n3. Creating HCD model...")
model = HCD(
    nodes=nodes,
    attrib=features,
    method='top_down',              # Top-down hierarchical detection
    ae_hidden_dims=[256, 128],      # Autoencoder hidden layer sizes
    comm_sizes=comm_sizes,          # Community sizes from step 2
    ae_operator='GATv2Conv',        # Graph attention operator
    dropout=0.2,
    normalize_input=True,
).to(DEVICE)

print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# 4. Train model
print("\n4. Training model...")
trainer = Trainer(
    model=model,
    X=train_set[0],
    A=train_set[1],
    validation_data=val_set,
    epochs=30,                      # Number of training epochs
    learning_rate=1e-3,             # Learning rate
    batch_size=32,                  # Batch size
    early_stopping=True,            # Enable early stopping
    patience=5,                     # Stop if no improvement for 5 epochs
    true_labels= train_set[2],  # For evaluation
    output_path=OUTPUT_PATH,           # Output directory
    verbose=True
)

output = trainer.fit(DEVICE)

# 5. View results
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

print("\nTraining Results:")
print(f"   Final training loss: {output.train_loss_history[-1]['Total Loss']:.4f}")

if output.performance_history:
    final_perf = output.performance_history[-1]
    print(f"\n   Performance Metrics (Last Epoch):")
    print(f"   Top Layer    - Homogeneity: {final_perf[0][0]:.4f}, Completeness: {final_perf[0][1]:.4f}, NMI: {final_perf[0][2]:.4f}, ARI: {final_perf[0][3]:.4f}")
    if len(final_perf) > 1:
        print(f"   Middle Layer - Homogeneity: {final_perf[1][0]:.4f}, Completeness: {final_perf[1][1]:.4f}, NMI: {final_perf[1][2]:.4f}, ARI: {final_perf[1][3]:.4f}")

print(f"\n   Predicted communities:")
print(f"   Top layer: {len(torch.unique(output.predicted_train['top']))} communities")
print(f"   Middle layer: {len(torch.unique(output.predicted_train['middle']))} communities")

print("\nModel trained successfully!")
print(f"   Access results via 'output' object")
print(f"   - output.predicted_train['top/middle']: Predicted labels")
print(f"   - output.probabilities: Community assignment probabilities")
print(f"   - output.train_loss_history: Training loss over epochs")

# 6. Save model for inference
MODEL_PATH = os.path.join(OUTPUT_PATH, 'trained_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'comm_sizes': comm_sizes,
    'config': {
        'method': 'top_down',
        'ae_hidden_dims': [256, 128],
        'ae_operator': 'GATv2Conv',
        'attn_heads': 1,  # Default value used in model creation
        'dropout': 0.2,
        'normalize_input': True,
        'normalize_outputs': True
    }
}, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

end_time = time.perf_counter()
file_path = "./1kgraph_numeric_stability_inference/time_of_execution.txt"

with open(file_path, "w") as file:
    elapsed_time = end_time - start_time
    file.write(f"Execution time: {elapsed_time:.4f} seconds")
    
print(f"File '{file_path}' created successfully.")

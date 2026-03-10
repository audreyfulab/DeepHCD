"""
Train DeepHCD on SUBSET of Converted Seurat RDS Data
This version ACTUALLY subsets the data properly!
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import time

from deephcd.model.model import HCD
from deephcd.model.train import Trainer
from deephcd.utils.utilities import compute_kappa
from deephcd.utils.utilities import get_input_graph

start_time = time.perf_counter()

# ============================================================================
# SUBSET CONFIGURATION - SET THESE!
# ============================================================================

USE_SUBSET = True           # ← Set to True to enable subsetting
SUBSET_SIZE = 500          # ← Number of cells you want (change this!)
SUBSET_METHOD = 'random'    # Options: 'random', 'stratified'
RANDOM_SEED = 42            # For reproducibility

# ============================================================================
# Configuration
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONVERTED_DATA_DIR = os.path.join(_SCRIPT_DIR, 'converted_data')
OUTPUT_PATH = os.path.join(_SCRIPT_DIR, 'deephcd_subset_training')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training hyperparameters (will auto-adjust based on subset size)
USE_PCA = True
N_PCS = 50
BUILD_KNN_IF_NO_GRAPH = True
K_NEIGHBORS = 30

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("="*80)
print("TRAINING DEEPHCD MODEL - SUBSET MODE")
print("="*80)
print(f"\nDevice: {DEVICE}")
print(f"Subset enabled: {USE_SUBSET}")
if USE_SUBSET:
    print(f"Target subset size: {SUBSET_SIZE} cells")
    print(f"Subset method: {SUBSET_METHOD}")

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading Data")
print("="*80)

def load_converted_data(data_dir):
    """Load data converted from RDS"""
    data = {}
    
    print("Loading expression matrix...")
    expr_path = os.path.join(data_dir, "expression_matrix.mtx")
    expression = mmread(expr_path).toarray()
    print(f"  Expression: {expression.shape}")
    
    genes = pd.read_csv(os.path.join(data_dir, "genes.csv"))['gene'].tolist()
    cells = pd.read_csv(os.path.join(data_dir, "cells.csv"))['cell'].tolist()
    print(f"  Genes: {len(genes)}, Cells: {len(cells)}")
    
    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
    print(f"  Metadata: {metadata.shape}")
    
    adj_path = os.path.join(data_dir, "adjacency_matrix.mtx")
    if os.path.exists(adj_path):
        print("Loading adjacency matrix (sparse)...")
        adjacency = mmread(adj_path).toarray()
        print(f"  Adjacency: {adjacency.shape}")
    else:
        adjacency = None
        print("  No adjacency matrix found")
    
    cluster_path = os.path.join(data_dir, "cluster_labels.csv")
    if os.path.exists(cluster_path):
        clusters = pd.read_csv(cluster_path)
        labels = clusters['cluster'].values
        n_clusters = len(np.unique(labels))
        print(f"  Cluster labels: {n_clusters} clusters")
    else:
        labels = None
        print("  No cluster labels")
    
    data['expression'] = expression
    data['genes'] = genes
    data['cells'] = cells
    data['metadata'] = metadata
    data['adjacency'] = adjacency
    data['labels'] = labels
    
    return data

data = load_converted_data(CONVERTED_DATA_DIR)

# ============================================================================
# STEP 1.5: APPLY SUBSETTING (THE CRITICAL STEP!)
# ============================================================================

print("\n" + "="*80)
print("STEP 1.5: Applying Subset")
print("="*80)

# Transpose if needed (ensure cells x genes)
if data['expression'].shape[0] < data['expression'].shape[1]:
    print("Transposing expression matrix to (cells x genes)")
    data['expression'] = data['expression'].T

total_cells = data['expression'].shape[0]
print(f"Original data: {total_cells} cells × {len(data['genes'])} genes")

if USE_SUBSET and total_cells > SUBSET_SIZE:
    print(f"\n SUBSETTING: {SUBSET_SIZE} / {total_cells} cells ({100*SUBSET_SIZE/total_cells:.1f}%)")
    print(f"Method: {SUBSET_METHOD}")
    
    np.random.seed(RANDOM_SEED)
    
    # ========================================================================
    # METHOD 1: Random Sampling
    # ========================================================================
    if SUBSET_METHOD == 'random':
        print("\nApplying random sampling...")
        indices = np.random.choice(total_cells, size=SUBSET_SIZE, replace=False)
        indices = np.sort(indices)  # Keep order for reproducibility
        print(f"  Selected {len(indices)} random cells")
    
    # ========================================================================
    # METHOD 2: Stratified Sampling (Balanced by Clusters)
    # ========================================================================
    elif SUBSET_METHOD == 'stratified' and data['labels'] is not None:
        print("\nApplying stratified sampling (balanced by clusters)...")
        
        labels = data['labels']
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        print(f"  Original: {n_clusters} clusters")
        
        indices_list = []
        for label in unique_labels:
            # Find all cells in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_size = len(cluster_indices)
            
            # Sample proportionally to cluster size
            n_from_cluster = max(1, int(SUBSET_SIZE * cluster_size / total_cells))
            
            # Don't sample more than available
            if n_from_cluster < cluster_size:
                sampled = np.random.choice(cluster_indices, size=n_from_cluster, replace=False)
            else:
                sampled = cluster_indices
            
            indices_list.append(sampled)
            print(f"    Cluster {label}: {len(sampled)}/{cluster_size} cells")
        
        # Combine all sampled indices
        indices = np.concatenate(indices_list)
        indices = np.sort(indices)
        print(f"\n  Total selected: {len(indices)} cells from {n_clusters} clusters")
    
    else:
        # Fallback to random if stratified requested but no labels
        print("\nStratified sampling requested but no labels available")
        print("Falling back to random sampling...")
        indices = np.random.choice(total_cells, size=SUBSET_SIZE, replace=False)
        indices = np.sort(indices)
        print(f"  Selected {len(indices)} random cells")
    
    # ========================================================================
    # APPLY THE SUBSET TO ALL DATA
    # ========================================================================
    print("\nApplying subset to data structures...")
    
    # Expression matrix
    print(f"  Expression: {data['expression'].shape} → ", end='')
    data['expression'] = data['expression'][indices, :]
    print(f"{data['expression'].shape}")
    
    # Cell names
    print(f"  Cells: {len(data['cells'])} → ", end='')
    data['cells'] = [data['cells'][i] for i in indices]
    print(f"{len(data['cells'])}")
    
    # Metadata
    print(f"  Metadata: {data['metadata'].shape} → ", end='')
    data['metadata'] = data['metadata'].iloc[indices].copy()
    print(f"{data['metadata'].shape}")
    
    # Adjacency matrix (if exists)
    if data['adjacency'] is not None:
        print(f"  Adjacency: {data['adjacency'].shape} → ", end='')
        # CRITICAL: Subset both rows AND columns
        data['adjacency'] = data['adjacency'][indices][:, indices]
        print(f"{data['adjacency'].shape}")
    
    # Labels (if exist)
    if data['labels'] is not None:
        print(f"  Labels: {len(data['labels'])} → ", end='')
        data['labels'] = data['labels'][indices]
        print(f"{len(data['labels'])}")
    
    print(f"\n SUBSET APPLIED SUCCESSFULLY")
    print(f"   New data size: {data['expression'].shape[0]} cells × {data['expression'].shape[1]} genes")

else:
    if not USE_SUBSET:
        print("\n  Subset disabled (USE_SUBSET=False)")
        print("   Using all data")
    else:
        print(f"\n  Requested subset size ({SUBSET_SIZE}) >= total cells ({total_cells})")
        print("   Using all data")

# ============================================================================
# STEP 2: Prepare Data
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Preparing Data")
print("="*80)

expression = data['expression']
n_cells, n_genes = expression.shape
print(f"Working with: {n_cells} cells × {n_genes} genes")

# PCA
if USE_PCA and n_genes > N_PCS:
    print(f"\nApplying PCA: {n_genes} genes → {N_PCS} PCs")
    pca = PCA(n_components=N_PCS, random_state=42)
    X = pca.fit_transform(expression)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained: {var_explained:.1%}")
else:
    print("\nUsing raw expression")
    X = expression

# Normalize
print("\nNormalizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"  Features normalized (mean=0, std=1)")

# Build or use adjacency
if data['adjacency'] is not None and data['adjacency'].shape[0] == n_cells:

    print("\nUsing provided adjacency matrix from Seurat...")
    A_graph, A = get_input_graph(X=X, method='KNN', K=K_NEIGHBORS)


else:
    print("\nBuilding KNN graph from PCA features...")
    A_graph, A = get_input_graph(
        X=X,                # numpy array (n_cells x n_pcs)
        method='KNN',
        K=K_NEIGHBORS,     
        metric='1-R^2'      
    )
 

print(f"  Graph: {A_graph.number_of_nodes()} nodes, {A_graph.number_of_edges()} edges")
print(f"  Adjacency density: {A.sum() / (n_cells**2):.4f}")

# --- Convert to tensor and add self-loops ---
A = torch.FloatTensor(A)
A = A + torch.eye(A.shape[0])
   # self-loops so every node attends to itself
A = torch.clamp(A, 0, 1)        # ensure binary after self-loop addition

print(f"  A tensor: {A.shape}, non-zero: {(A > 0).sum().item()}")
# Convert to PyTorch tensors
X = torch.FloatTensor(X)
A = torch.FloatTensor(A)

nodes, features = X.shape

print(f"\nPrepared tensors:")
print(f"  X: {X.shape}")
print(f"  A: {A.shape}")

# Verify the subset worked!
assert nodes == SUBSET_SIZE or (not USE_SUBSET or total_cells <= SUBSET_SIZE), \
    f"Subset failed! Expected {SUBSET_SIZE} nodes, got {nodes}"

# Process labels
true_labels = None
if data['labels'] is not None:
    labels = data['labels']
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    print(f"\nProcessing labels: {n_clusters} clusters")
    
    if n_clusters > 20:
        from sklearn.cluster import KMeans
        n_top = max(5, n_clusters // 4)
        print(f"  Creating 2-level hierarchy: {n_top} (top) / {n_clusters} (middle)")
        
        kmeans_top = KMeans(n_clusters=n_top, random_state=42, n_init=10)
        labels_top = kmeans_top.fit_predict(X.numpy())
        
        true_labels = [labels_top, labels]
    else:
        print(f"  Using single level: {n_clusters} clusters")
        true_labels = [labels]

# ============================================================================
# STEP 3: Estimate Community Sizes
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Estimating Community Sizes")
print("="*80)

try:
    comm_middle, comm_top = compute_kappa(X, A, method='bethe_hessian', verbose=True)
    comm_sizes = [comm_middle, comm_top]
    print(f"Bethe-Hessian: Top={comm_top}, Middle={comm_middle}")
except Exception as e:
    print(f"Bethe-Hessian failed: {e}")
    comm_top = max(10, nodes // 200)
    comm_middle = max(20, nodes // 50)
    comm_sizes = [comm_middle, comm_top]
    print(f"Using heuristic: Top={comm_top}, Middle={comm_middle}")

# ============================================================================
# STEP 4: Create Model & Auto-Adjust Hyperparameters
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Creating Model")
print("="*80)

# Auto-adjust hyperparameters based on subset size
if nodes < 1000:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 200
    print(f"Small dataset (<1K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
elif nodes < 5000:
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128
    EPOCHS = 150
    print(f"Medium dataset (1-5K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
elif nodes < 10000:
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 256
    EPOCHS = 100
    print(f"Large dataset (5-10K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
else:
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 512
    EPOCHS = 100
    print(f"Very large dataset (>10K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")

EARLY_STOPPING = True
PATIENCE = 10

model = HCD(
    nodes=nodes,
    attrib=features,
    method='top_down',
    ae_hidden_dims=[128, 64, 32],
    ll_hidden_dims=[32, 32],
    comm_sizes=comm_sizes,
    ae_operator='GATv2Conv',
    comm_operator='Linear',
    dropout=0.3,
    use_output_layers=True,
    normalize_input=True,
    normalize_outputs=True,
    ae_attn_heads=2
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} parameters")

# ============================================================================
# STEP 5: Train
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Training Model")
print("="*80)

trainer = Trainer(
    model=model,
    X=X,
    A=A,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    gamma=1.0,
    delta=1.0,
    _lambda=[1.0, 1.0],
    graph_resolutions=[1.0, 1.0],
    k=len(comm_sizes),
    early_stopping=EARLY_STOPPING,
    patience=PATIENCE,
    use_batch_learning=True,
    true_labels=true_labels,
    output_path=OUTPUT_PATH,
    save_output=True,
    use_logging=True,
    log_to_file=True,
    verbose=True
)

print("Starting training...\n")
output = trainer.fit(DEVICE)

# ============================================================================
# STEP 6: Save Results
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

if output.train_loss_history:
    final_loss = output.train_loss_history[-1]['Total Loss']
    print(f"\nFinal training loss: {final_loss:.4f}")

if output.performance_history:
    final_perf = None
    for perf in reversed(output.performance_history):
        if perf is not None:
            final_perf = perf
            break
    
    if final_perf:
        print(f"\nPerformance Metrics:")
        for i, perf in enumerate(final_perf):
            if perf is not None:
                layer = 'Top' if i == 0 else 'Middle'
                print(f"  {layer}: H={perf[0]:.3f}, C={perf[1]:.3f}, NMI={perf[2]:.3f}, ARI={perf[3]:.3f}")

# Save model
MODEL_PATH = os.path.join(OUTPUT_PATH, 'trained_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'comm_sizes': comm_sizes,
    'subset_size': nodes,
    'config': {
        'method': 'top_down',
        'ae_hidden_dims': [128, 64, 32],
        'comm_operator': 'Linear',
        'dropout': 0.3
    }
}, MODEL_PATH)
print(f"\n✓ Model saved: {MODEL_PATH}")

# Save predictions
if hasattr(output, 'predicted_train'):
    pred_path = os.path.join(OUTPUT_PATH, 'predictions.csv')
    top_preds = output.predicted_train['top'].cpu().numpy() if 'top' in output.predicted_train else None
    mid_preds = output.predicted_train['middle'].cpu().numpy() if 'middle' in output.predicted_train else None
    n_preds = len(top_preds) if top_preds is not None else (len(mid_preds) if mid_preds is not None else len(data['cells']))
    cells_for_pred = data['cells'][:n_preds]
    pred_df = pd.DataFrame({
        'cell': cells_for_pred,
        'top_cluster': top_preds if top_preds is not None else [None] * n_preds,
        'middle_cluster': mid_preds if mid_preds is not None else [None] * n_preds,
    })
    pred_df.to_csv(pred_path, index=False)
    print(f"✓ Predictions saved: {pred_path}")

# Save timing
end_time = time.perf_counter()
elapsed = end_time - start_time

time_file = os.path.join(OUTPUT_PATH, "execution_time.txt")
with open(time_file, "w") as f:
    f.write(f"Execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)\n")
    f.write(f"Subset size: {nodes} cells\n")
    f.write(f"Subset method: {SUBSET_METHOD}\n")

print(f"✓ Timing saved: {time_file}")

# ============================================================================
# HEATMAPS
# ============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not hasattr(output, 'predicted_train'):
        pred_df = pd.read_csv(pred_path)
    else:
        _top = output.predicted_train['top'].cpu().numpy() if 'top' in output.predicted_train else None
        _mid = output.predicted_train['middle'].cpu().numpy() if 'middle' in output.predicted_train else None
        _n = len(_top) if _top is not None else (len(_mid) if _mid is not None else len(data['cells']))
        pred_df = pd.DataFrame({
            'cell': data['cells'][:_n],
            'top_cluster': _top if _top is not None else [None] * _n,
            'middle_cluster': _mid if _mid is not None else [None] * _n,
        })

    X_np = X.numpy()[:_n]  # align with predicted cells

    # ── 1. Cell-cell correlation heatmap sorted by cluster ────────────────
    sorted_df = pred_df.dropna(subset=['top_cluster']).sort_values(['top_cluster', 'middle_cluster'])
    sorted_idx = sorted_df.index.to_numpy()
    X_sorted = X_np[sorted_idx]

    # Pearson correlation between every pair of cells
    corr_matrix = np.corrcoef(X_sorted)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    top_clusters = sorted_df['top_cluster'].values
    boundaries = np.where(np.diff(top_clusters))[0] + 1
    for b in boundaries:
        ax.axhline(b, color='black', linewidth=1.2, linestyle='--')
        ax.axvline(b, color='black', linewidth=1.2, linestyle='--')
    ax.set_title('Cell-Cell Correlation (sorted by Top Cluster)')
    ax.set_xlabel('Cells')
    ax.set_ylabel('Cells')
    plt.tight_layout()
    heatmap1_path = os.path.join(OUTPUT_PATH, 'heatmap_cell_correlation.png')
    plt.savefig(heatmap1_path, dpi=150)
    plt.close()
    print(f"✓ Cell-cell correlation heatmap saved: {heatmap1_path}")

    # ── 2. Inter-cluster mean correlation heatmap ─────────────────────────
    for cluster_level, cluster_col in [('top', 'top_cluster'), ('middle', 'middle_cluster')]:
        co = pred_df.dropna(subset=[cluster_col])
        if len(co) == 0:
            continue
        cluster_ids = sorted(co[cluster_col].astype(int).unique())
        # Mean expression vector per cluster
        cluster_means = np.stack([
            X_np[co[co[cluster_col].astype(int) == c].index].mean(axis=0)
            for c in cluster_ids
        ])
        inter_corr = np.corrcoef(cluster_means)
        fig, ax = plt.subplots(figsize=(max(6, len(cluster_ids)), max(5, len(cluster_ids) - 1)))
        sns.heatmap(
            inter_corr,
            ax=ax,
            cmap='coolwarm',
            center=0,
            vmin=-1, vmax=1,
            annot=True, fmt='.2f',
            xticklabels=[f'C{c}' for c in cluster_ids],
            yticklabels=[f'C{c}' for c in cluster_ids],
            cbar_kws={'label': 'Pearson Correlation'}
        )
        ax.set_title(f'Inter-Cluster Correlation ({cluster_level.capitalize()} level)')
        plt.tight_layout()
        heatmap2_path = os.path.join(OUTPUT_PATH, f'heatmap_cluster_corr_{cluster_level}.png')
        plt.savefig(heatmap2_path, dpi=150)
        plt.close()
        print(f"✓ Cluster correlation heatmap saved: {heatmap2_path}")

except Exception as e:
    print(f"⚠ Heatmap generation failed: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Trained on {nodes} / {total_cells} cells ({100*nodes/total_cells:.1f}%)")
print(f"✓ Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"✓ Results: {OUTPUT_PATH}")
print("="*80)
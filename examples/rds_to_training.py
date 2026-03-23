"""
RDS to DeepHCD Training Pipeline
Reads a Seurat RDS file directly, converts it to Python-readable sparse formats,
then trains DeepHCD — mirroring single_cell_training.py end-to-end.
"""

import os
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, issparse
from scipy.io import mmwrite, mmread
import time

from deephcd.model.model import HCD
from deephcd.model.train import Trainer
from deephcd.utils.utilities import compute_kappa
from deephcd.utils.utilities import get_input_graph

start_time = time.perf_counter()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your Seurat RDS file
RDS_PATH = '/Users/jordandavis/Documents/DeepHCD_copy/14_16_finished_processing.Rds'

# Where to write the converted data files (mirrors converted_data/)
CONVERTED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'converted_data_from_rds_2'
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'deephcd_rds_training'
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Subset configuration
USE_SUBSET = True
SUBSET_SIZE = 500
SUBSET_METHOD = 'random'   # 'random' or 'stratified'
RANDOM_SEED = 42

# Which Seurat cluster column to use as true labels
# Common options: 'seurat_clusters', 'RNA_snn_res.0.37', etc.
CLUSTER_COLUMN = 'seurat_clusters'

# PCA / graph settings
N_PCS = 50       # PCA components for dimension reduction
K_NEIGHBORS = 30

os.makedirs(CONVERTED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 80)
print("RDS → DEEPHCD TRAINING PIPELINE")
print("=" * 80)
print(f"\nDevice: {DEVICE}")
print(f"RDS file: {RDS_PATH}")
print(f"Converted data dir: {CONVERTED_DATA_DIR}")

# ============================================================================
# STEP 1: Read RDS and convert to sparse Python formats
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: Reading RDS File and Converting to Sparse Format")
print("=" * 80)


def convert_rds_to_sparse(rds_path: str, out_dir: str, cluster_col: str = 'seurat_clusters'):
    """
    Load a Seurat RDS object via rpy2 and export to Python-readable sparse files:
      - expression_matrix.npz   (scipy sparse, cells × genes)
      - adjacency_matrix.npz    (scipy sparse, SNN graph)
      - genes.csv
      - cells.csv
      - metadata.csv
      - cluster_labels.csv      (from Seurat cluster assignments)
      - pca_embeddings.csv
      - umap_embeddings.csv

    Returns a dict with the same keys as load_converted_data() in single_cell_training.py.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr

    base = importr('base')
    Matrix = importr('Matrix')

    print("Loading RDS file (this may take a while for large objects)...")
    seurat_obj = r(f'readRDS("{rds_path}")')
    r_obj_name = 'seurat_obj_tmp'
    ro.globalenv[r_obj_name] = seurat_obj

    # ---- Gene and cell names ------------------------------------------------
    print("Extracting gene and cell names...")
    genes = list(r(f'rownames({r_obj_name})'))
    cells = list(r(f'colnames({r_obj_name})'))
    n_genes = len(genes)
    n_cells = len(cells)
    print(f"  Genes: {n_genes}, Cells: {n_cells}")

    pd.DataFrame({'gene': genes}).to_csv(os.path.join(out_dir, 'genes.csv'), index=False)
    pd.DataFrame({'cell': cells}).to_csv(os.path.join(out_dir, 'cells.csv'), index=False)

    # ---- Expression matrix (sparse) -----------------------------------------
    # Seurat v5 uses layers instead of slots — try in the same order as the R script
    print("Extracting expression matrix (Seurat v5 layer API)...")
    expr_mat_loaded = False
    for layer in ('data', 'counts'):
        try:
            r(f'expr_mat <- GetAssayData({r_obj_name}, assay="RNA", layer="{layer}")')
            _ = r('dim(expr_mat)')   # force evaluation; will error if NULL
            print(f"  Loaded layer='{layer}'")
            expr_mat_loaded = True
            break
        except Exception:
            pass
    if not expr_mat_loaded:
        # Last-resort fallback to direct slot access (Seurat v4 style)
        r(f'expr_mat <- {r_obj_name}[["RNA"]]$data')
        print("  Loaded via direct $data accessor (fallback)")

    # Normalise to dgCMatrix so @i/@p/@x slots are guaranteed
    r('if (!inherits(expr_mat, "dgCMatrix")) { expr_mat <- as(expr_mat, "CsparseMatrix") }')
    mat_class = list(r('class(expr_mat)'))[0]
    print(f"  Matrix class after normalisation: {mat_class}")

    r('expr_dims <- dim(expr_mat)')
    dims = list(r('expr_dims'))
    n_g, n_c = int(dims[0]), int(dims[1])

    # dgCMatrix stores data column-major (CSC): @i=row indices, @p=col ptrs, @x=values
    rows_r   = np.array(r('expr_mat@i'), dtype=np.int32)      # already 0-based in R's C layer
    col_ptr  = np.array(r('expr_mat@p'), dtype=np.int64)
    vals     = np.array(r('expr_mat@x'), dtype=np.float32)

    col_indices = np.repeat(np.arange(n_c), np.diff(col_ptr))
    # Transpose to cells × genes (swap row/col)
    expr_sparse = csr_matrix(
        (vals, (col_indices, rows_r)),
        shape=(n_c, n_g)
    )
    expr_npz_path = os.path.join(out_dir, 'expression_matrix.npz')
    save_npz(expr_npz_path, expr_sparse)
    print(f"  Saved sparse expression matrix: {expr_sparse.shape}  → {expr_npz_path}")

    # MTX copy for compatibility with single_cell_training.py
    expr_mtx_path = os.path.join(out_dir, 'expression_matrix.mtx')
    mmwrite(expr_mtx_path, expr_sparse)
    print(f"  Saved MTX expression matrix: {expr_mtx_path}")

    # ---- Adjacency / SNN graph (sparse) -------------------------------------
    # Mirrors the R script: prefer RNA_snn, fall back to first available graph
    print("Extracting SNN adjacency matrix...")
    graph_names = list(r(f'names({r_obj_name}@graphs)'))
    print(f"  Available graphs: {graph_names}")
    snn_name = 'RNA_snn' if 'RNA_snn' in graph_names else (graph_names[0] if graph_names else None)

    if snn_name:
        print(f"  Using graph: {snn_name}")
        r(f'adj_mat <- {r_obj_name}@graphs${snn_name}')

        # Seurat Graph objects inherit from dgCMatrix but may need coercion
        r('if (!inherits(adj_mat, "dgCMatrix")) { adj_mat <- as(adj_mat, "CsparseMatrix") }')
        adj_class = list(r('class(adj_mat)'))[0]
        print(f"  Adjacency class after normalisation: {adj_class}")

        adj_dim = int(r('nrow(adj_mat)')[0])
        adj_rows    = np.array(r('adj_mat@i'), dtype=np.int32)
        adj_col_ptr = np.array(r('adj_mat@p'), dtype=np.int64)
        adj_vals    = np.array(r('adj_mat@x'), dtype=np.float32)

        adj_col_indices = np.repeat(np.arange(adj_dim), np.diff(adj_col_ptr))
        adj_sparse = csr_matrix(
            (adj_vals, (adj_rows, adj_col_indices)),
            shape=(adj_dim, adj_dim)
        )

        adj_npz_path = os.path.join(out_dir, 'adjacency_matrix.npz')
        save_npz(adj_npz_path, adj_sparse)
        print(f"  Saved sparse adjacency matrix: {adj_sparse.shape}  → {adj_npz_path}")

        adj_mtx_path = os.path.join(out_dir, 'adjacency_matrix.mtx')
        mmwrite(adj_mtx_path, adj_sparse)
        print(f"  Saved MTX adjacency matrix: {adj_mtx_path}")
    else:
        adj_sparse = None
        print("  No graphs found in Seurat object — Python will build KNN graph")

    # ---- Metadata -----------------------------------------------------------
    print("Extracting metadata...")
    meta_cols = list(r(f'colnames({r_obj_name}@meta.data)'))
    meta_dict = {'cell': cells}
    for col in meta_cols:
        try:
            vals_col = list(r(f'{r_obj_name}@meta.data${col}'))
            meta_dict[col] = vals_col
        except Exception:
            pass
    metadata = pd.DataFrame(meta_dict).set_index('cell')
    metadata.to_csv(os.path.join(out_dir, 'metadata.csv'))
    print(f"  Metadata: {metadata.shape}  cols: {list(metadata.columns)}")

    # ---- Cluster labels -----------------------------------------------------
    # Search order mirrors the R script, with seurat_clusters first
    cluster_search = [cluster_col, 'seurat_clusters', 'RNA_snn_res.0.37',
                      'RNA_snn_res.0.5', 'RNA_snn_res.1', 'clusters', 'celltype', 'cell_type']
    found_col = next((c for c in cluster_search if c in metadata.columns), None)

    if found_col:
        if found_col != cluster_col:
            print(f"  '{cluster_col}' not found — using '{found_col}' instead")
        else:
            print(f"Extracting cluster labels from column: '{found_col}'...")

        raw_labels = metadata[found_col].values
        # Convert factor/string labels to integers (same logic as R script)
        try:
            int_labels = pd.to_numeric(raw_labels).astype(int)
        except (ValueError, TypeError):
            unique_vals = sorted(set(raw_labels), key=str)
            label_map = {v: i for i, v in enumerate(unique_vals)}
            int_labels = np.array([label_map[v] for v in raw_labels])

        cluster_df = pd.DataFrame({'cell': cells, 'cluster': int_labels})
        cluster_path = os.path.join(out_dir, 'cluster_labels.csv')
        cluster_df.to_csv(cluster_path, index=False)
        print(f"  Saved cluster labels: {len(np.unique(int_labels))} clusters → {cluster_path}")
    else:
        int_labels = None
        available = [c for c in metadata.columns if 'cluster' in c.lower() or 'res' in c.lower()]
        print(f"  WARNING: No cluster column found. Available options: {available}")

    # ---- Dimensionality reductions (PCA, UMAP, etc.) ------------------------
    # Use Embeddings() like the R script — more reliable than @reductions$x@cell.embeddings
    reduction_names = list(r(f'names({r_obj_name}@reductions)'))
    print(f"  Available reductions: {reduction_names}")
    for red_name in reduction_names:
        try:
            emb = np.array(r(f'Embeddings({r_obj_name}, reduction="{red_name}")'))
            if emb.shape[1] > 100:
                print(f"  Skipped {red_name} (too many dims: {emb.shape[1]})")
                continue
            emb_df = pd.DataFrame(emb, index=cells,
                                  columns=[f'{red_name.upper()}_{i+1}' for i in range(emb.shape[1])])
            emb_path = os.path.join(out_dir, f'{red_name}_embeddings.csv')
            emb_df.to_csv(emb_path)
            print(f"  Saved {red_name} embeddings: {emb_df.shape}  → {emb_path}")
        except Exception as e:
            print(f"  {red_name} embeddings not available: {e}")

    # ---- Object summary -----------------------------------------------------
    summary_path = os.path.join(out_dir, 'object_info.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Source RDS: {rds_path}\n")
        f.write(f"Number of cells: {n_cells}\n")
        f.write(f"Number of genes: {n_genes}\n")
        f.write(f"Expression sparsity: {1 - expr_sparse.nnz / (n_cells * n_genes):.4f}\n")
        f.write(f"Cluster column used: {cluster_col}\n")
        f.write(f"Graphs found: {graph_names}\n")
    print(f"  Saved object summary: {summary_path}")

    return {
        'expression': expr_sparse,    # scipy sparse csr (cells × genes)
        'adjacency': adj_sparse,       # scipy sparse csr or None
        'genes': genes,
        'cells': cells,
        'metadata': metadata,
        'labels': int_labels,
    }


data = convert_rds_to_sparse(RDS_PATH, CONVERTED_DATA_DIR, cluster_col=CLUSTER_COLUMN)

# ============================================================================
# STEP 1.5: Apply Subsetting
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1.5: Applying Subset")
print("=" * 80)

expr = data['expression']

# Ensure cells × genes orientation
if expr.shape[0] < expr.shape[1]:
    print("Transposing expression matrix to (cells × genes)")
    expr = expr.T

total_cells = expr.shape[0]
print(f"Original data: {total_cells} cells × {len(data['genes'])} genes")

if USE_SUBSET and total_cells > SUBSET_SIZE:
    print(f"\nSUBSETTING: {SUBSET_SIZE} / {total_cells} cells ({100*SUBSET_SIZE/total_cells:.1f}%)")
    np.random.seed(RANDOM_SEED)

    if SUBSET_METHOD == 'random':
        indices = np.sort(np.random.choice(total_cells, size=SUBSET_SIZE, replace=False))

    elif SUBSET_METHOD == 'stratified' and data['labels'] is not None:
        labels_all = data['labels']
        unique_labels = np.unique(labels_all)
        indices_list = []
        for label in unique_labels:
            cluster_indices = np.where(labels_all == label)[0]
            n_from = max(1, int(SUBSET_SIZE * len(cluster_indices) / total_cells))
            if n_from < len(cluster_indices):
                sampled = np.random.choice(cluster_indices, size=n_from, replace=False)
            else:
                sampled = cluster_indices
            indices_list.append(sampled)
            print(f"  Cluster {label}: {len(sampled)}/{len(cluster_indices)} cells")
        indices = np.sort(np.concatenate(indices_list))

    else:
        print("Falling back to random sampling...")
        indices = np.sort(np.random.choice(total_cells, size=SUBSET_SIZE, replace=False))

    expr = expr[indices, :]
    data['cells'] = [data['cells'][i] for i in indices]
    data['metadata'] = data['metadata'].iloc[indices].copy()
    if data['adjacency'] is not None:
        data['adjacency'] = data['adjacency'][indices][:, indices]
    if data['labels'] is not None:
        data['labels'] = data['labels'][indices]

    print(f"\nSUBSET APPLIED: {expr.shape[0]} cells × {expr.shape[1]} genes")

else:
    print("Using full dataset")

# ============================================================================
# STEP 2: Prepare Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Preparing Data")
print("=" * 80)

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

n_cells, n_genes = expr.shape
print(f"Working with: {n_cells} cells × {n_genes} genes")

if n_genes > N_PCS:
    print(f"\nApplying TruncatedSVD (sparse PCA): {n_genes} genes → {N_PCS} PCs")
    svd = TruncatedSVD(n_components=N_PCS, random_state=42)
    X = svd.fit_transform(expr)
    print(f"  Variance explained: {svd.explained_variance_ratio_.sum():.1%}")
    print(f"  PCA embedding shape: {X.shape}")
else:
    X = expr.toarray() if issparse(expr) else expr

print("\nNormalizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)

if data['adjacency'] is not None and data['adjacency'].shape[0] == n_cells:
    print("\nUsing Seurat SNN adjacency matrix (correlation-based SNN — no KNN rebuild)...")
    A_np = (data['adjacency'].toarray() if issparse(data['adjacency'])
            else data['adjacency']).astype(np.float32)
    n_edges = int((A_np > 0).sum())
    print(f"  {n_cells} nodes, {n_edges} edges")
    print(f"  Adjacency density: {A_np.sum() / (n_cells**2):.4f}")
else:
    print("\nNo adjacency matrix found — building KNN graph from PCA features...")
    A_graph, A_np = get_input_graph(X=X, method='KNN', K=K_NEIGHBORS, metric='1-R^2')
    print(f"  Graph: {A_graph.number_of_nodes()} nodes, {A_graph.number_of_edges()} edges")
    print(f"  Adjacency density: {A_np.sum() / (n_cells**2):.4f}")

A = torch.clamp(torch.FloatTensor(A_np) + torch.eye(n_cells), 0, 1)
X = torch.FloatTensor(X)
nodes, features = X.shape

print(f"\nPrepared tensors:")
print(f"  X: {X.shape}")
print(f"  A: {A.shape}")

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
        labels_top = KMeans(n_clusters=n_top, random_state=42, n_init=10).fit_predict(X.numpy())
        true_labels = [labels_top, labels]
    else:
        print(f"  Using single level: {n_clusters} clusters")
        true_labels = [labels]

# ============================================================================
# STEP 3: Estimate Community Sizes
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Estimating Community Sizes")
print("=" * 80)

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
# STEP 4: Create Model
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Creating Model")
print("=" * 80)

if nodes < 1000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-3, 64, 200
    print(f"Small dataset (<1K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
elif nodes < 5000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-4, 128, 150
    print(f"Medium dataset (1-5K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
elif nodes < 10000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-5, 256, 100
    print(f"Large dataset (5-10K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
else:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-5, 512, 100
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

print("\n" + "=" * 80)
print("STEP 5: Training Model")
print("=" * 80)

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

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

if output.train_loss_history:
    print(f"\nFinal training loss: {output.train_loss_history[-1]['Total Loss']:.4f}")

if output.performance_history:
    final_perf = next((p for p in reversed(output.performance_history) if p is not None), None)
    if final_perf:
        print("\nPerformance Metrics:")
        for i, perf in enumerate(final_perf):
            if perf is not None:
                layer = 'Top' if i == 0 else 'Middle'
                print(f"  {layer}: H={perf[0]:.3f}, C={perf[1]:.3f}, NMI={perf[2]:.3f}, ARI={perf[3]:.3f}")

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
print(f"\nModel saved: {MODEL_PATH}")

if hasattr(output, 'predicted_train'):
    top_preds = output.predicted_train['top'].cpu().numpy() if 'top' in output.predicted_train else None
    mid_preds = output.predicted_train['middle'].cpu().numpy() if 'middle' in output.predicted_train else None
    n_preds = len(top_preds) if top_preds is not None else (len(mid_preds) if mid_preds is not None else len(data['cells']))
    pred_df = pd.DataFrame({
        'cell': data['cells'][:n_preds],
        'top_cluster': top_preds if top_preds is not None else [None] * n_preds,
        'middle_cluster': mid_preds if mid_preds is not None else [None] * n_preds,
    })
    pred_path = os.path.join(OUTPUT_PATH, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")

end_time = time.perf_counter()
elapsed = end_time - start_time
time_file = os.path.join(OUTPUT_PATH, 'execution_time.txt')
with open(time_file, 'w') as f:
    f.write(f"Execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)\n")
    f.write(f"Subset size: {nodes} cells\n")
    f.write(f"Subset method: {SUBSET_METHOD}\n")
    f.write(f"RDS source: {RDS_PATH}\n")
print(f"Timing saved: {time_file}")

# ============================================================================
# HEATMAPS
# ============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    _top = output.predicted_train['top'].cpu().numpy() if 'top' in output.predicted_train else None
    _mid = output.predicted_train['middle'].cpu().numpy() if 'middle' in output.predicted_train else None
    _n = len(_top) if _top is not None else (len(_mid) if _mid is not None else len(data['cells']))
    pred_df = pd.DataFrame({
        'cell': data['cells'][:_n],
        'top_cluster': _top if _top is not None else [None] * _n,
        'middle_cluster': _mid if _mid is not None else [None] * _n,
    })

    X_np = X.numpy()[:_n]

    sorted_df = pred_df.dropna(subset=['top_cluster']).sort_values(['top_cluster', 'middle_cluster'])
    sorted_idx = sorted_df.index.to_numpy()
    corr_matrix = np.corrcoef(X_np[sorted_idx])

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Pearson Correlation'})
    top_clusters = sorted_df['top_cluster'].values
    for b in np.where(np.diff(top_clusters))[0] + 1:
        ax.axhline(b, color='black', linewidth=1.2, linestyle='--')
        ax.axvline(b, color='black', linewidth=1.2, linestyle='--')
    ax.set_title('Cell-Cell Correlation (sorted by Top Cluster)')
    ax.set_xlabel('Cells')
    ax.set_ylabel('Cells')
    plt.tight_layout()
    heatmap1_path = os.path.join(OUTPUT_PATH, 'heatmap_cell_correlation.png')
    plt.savefig(heatmap1_path, dpi=150)
    plt.close()
    print(f"Cell-cell correlation heatmap saved: {heatmap1_path}")

    for cluster_level, cluster_col in [('top', 'top_cluster'), ('middle', 'middle_cluster')]:
        co = pred_df.dropna(subset=[cluster_col])
        if len(co) == 0:
            continue
        cluster_ids = sorted(co[cluster_col].astype(int).unique())
        cluster_means = np.stack([
            X_np[co[co[cluster_col].astype(int) == c].index].mean(axis=0)
            for c in cluster_ids
        ])
        inter_corr = np.corrcoef(cluster_means)
        fig, ax = plt.subplots(figsize=(max(6, len(cluster_ids)), max(5, len(cluster_ids) - 1)))
        sns.heatmap(inter_corr, ax=ax, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    annot=True, fmt='.2f',
                    xticklabels=[f'C{c}' for c in cluster_ids],
                    yticklabels=[f'C{c}' for c in cluster_ids],
                    cbar_kws={'label': 'Pearson Correlation'})
        ax.set_title(f'Inter-Cluster Correlation ({cluster_level.capitalize()} level)')
        plt.tight_layout()
        heatmap2_path = os.path.join(OUTPUT_PATH, f'heatmap_cluster_corr_{cluster_level}.png')
        plt.savefig(heatmap2_path, dpi=150)
        plt.close()
        print(f"Cluster correlation heatmap saved: {heatmap2_path}")

except Exception as e:
    print(f"Heatmap generation failed: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Trained on {nodes} / {total_cells} cells ({100*nodes/total_cells:.1f}%)")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Converted data: {CONVERTED_DATA_DIR}")
print(f"Results: {OUTPUT_PATH}")
print("=" * 80)

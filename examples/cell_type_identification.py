"""
Drosophila scRNA-seq: Epithelial and Glial Cell Type Identification
===================================================================
Trains DeepHCD with exactly 2 forced top-level clusters, then scores each
cluster against known Drosophila marker genes to assign epithelial / glial
labels. Outputs annotated CSVs, a marker heatmap, and a UMAP plot.

Usage (local):
    python -u cell_type_identification.py

Usage (HPCC single-node):
    DEEPHCD_DATA_DIR=/path/to/converted_data \\
    DEEPHCD_OUTPUT_DIR=/path/to/output \\
    python -u cell_type_identification.py
"""

import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import torch
from scipy.io import mmread
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from deephcd.model.model import HCD
try:
    from deephcd.model.model import forward_timing, reset_forward_timing
except ImportError:
    forward_timing = {'gate_encoder': 0.0, 'gate_decoder': 0.0, 'clustering': 0.0, 'calls': 0}
    def reset_forward_timing(): pass
from deephcd.model.train import Trainer
from deephcd.utils.utilities import compute_kappa, get_input_graph

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_SUBSET    = True
SUBSET_SIZE   = 500
SUBSET_METHOD = 'stratified'   # 'random' | 'stratified'
RANDOM_SEED   = 42

_SCRIPT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERTED_DATA_DIR = os.environ.get('DEEPHCD_DATA_DIR',
                         os.path.join(_SCRIPT_DIR, 'converted_data'))
OUTPUT_PATH        = os.environ.get('DEEPHCD_OUTPUT_DIR',
                         os.path.join(_SCRIPT_DIR, 'cell_type_output'))
os.makedirs(OUTPUT_PATH, exist_ok=True)

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
N_PCS       = 50
K_NEIGHBORS = 30

# Drosophila marker genes for cell type scoring
GLIAL_MARKERS      = ['repo', 'gcm', 'Sox100B', 'wrapper', 'Eaat1']
EPITHELIAL_MARKERS = ['shg', 'crb', 'arm', 'dlg1', 'scrib']

EARLY_STOPPING = True
PATIENCE       = 10

# ============================================================================
# TIMING / MEMORY INFRASTRUCTURE
# ============================================================================

_step_times = {}

class StepTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        tracemalloc.clear_traces()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t0
        _, peak = tracemalloc.get_traced_memory()
        _step_times[self.name] = (elapsed, peak / 1024 / 1024)
        print(f"  [{self.name}] done in {_fmt(elapsed)}", flush=True)

def _fmt(seconds):
    if seconds >= 60:
        return f"{seconds/60:.1f} min"
    return f"{seconds:.2f}s"

tracemalloc.start()
start_time = time.perf_counter()

print("=" * 80)
print("DEEPHCD CELL TYPE IDENTIFICATION: EPITHELIAL vs GLIAL")
print("=" * 80)
print(f"\nDevice: {DEVICE}")
print(f"Subset: {USE_SUBSET} ({SUBSET_SIZE} cells, {SUBSET_METHOD})")
print(f"\nGlial markers:      {GLIAL_MARKERS}")
print(f"Epithelial markers: {EPITHELIAL_MARKERS}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: Loading Data")
print("=" * 80)

def load_converted_data(data_dir):
    data = {}

    print("Loading expression matrix...", flush=True)
    expression = mmread(os.path.join(data_dir, "expression_matrix.mtx")).tocsr()
    print(f"  Expression: {expression.shape}", flush=True)

    genes = pd.read_csv(os.path.join(data_dir, "genes.csv"))['gene'].tolist()
    cells = pd.read_csv(os.path.join(data_dir, "cells.csv"))['cell'].tolist()
    print(f"  Genes: {len(genes)}, Cells: {len(cells)}", flush=True)

    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
    print(f"  Metadata: {metadata.shape}", flush=True)

    adj_path = os.path.join(data_dir, "adjacency_matrix.mtx")
    if os.path.exists(adj_path):
        adjacency = mmread(adj_path).tocsr()
        print(f"  Adjacency: {adjacency.shape}", flush=True)
    else:
        adjacency = None
        print("  No adjacency matrix found", flush=True)

    cluster_path = os.path.join(data_dir, "cluster_labels.csv")
    if os.path.exists(cluster_path):
        clusters = pd.read_csv(cluster_path)
        labels = clusters['cluster'].values
        print(f"  Cluster labels: {len(np.unique(labels))} clusters", flush=True)
    else:
        labels = None
        print("  No cluster labels", flush=True)

    umap_path = os.path.join(data_dir, "umap_embeddings.csv")
    if os.path.exists(umap_path):
        data['umap'] = pd.read_csv(umap_path, index_col=0)
        print(f"  UMAP embeddings: {data['umap'].shape}", flush=True)
    else:
        data['umap'] = None
        print("  No UMAP embeddings found", flush=True)

    data['expression'] = expression
    data['genes']      = genes
    data['cells']      = cells
    data['metadata']   = metadata
    data['adjacency']  = adjacency
    data['labels']     = labels
    return data

with StepTimer("Step 1: Load Data"):
    data = load_converted_data(CONVERTED_DATA_DIR)

# ============================================================================
# STEP 1.5: SUBSET
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1.5: Applying Subset")
print("=" * 80)

with StepTimer("Step 1.5: Subset"):
    expression = data['expression']

    # Ensure cells × genes orientation
    if expression.shape[0] < expression.shape[1]:
        print("Transposing expression matrix to (cells × genes)")
        expression = expression.T

    total_cells = expression.shape[0]
    print(f"Original data: {total_cells} cells × {len(data['genes'])} genes")

    if USE_SUBSET and total_cells > SUBSET_SIZE:
        print(f"\n SUBSETTING: {SUBSET_SIZE} / {total_cells} cells ({100*SUBSET_SIZE/total_cells:.1f}%)")
        np.random.seed(RANDOM_SEED)

        if SUBSET_METHOD == 'random':
            indices = np.sort(np.random.choice(total_cells, size=SUBSET_SIZE, replace=False))
        elif SUBSET_METHOD == 'stratified' and data['labels'] is not None:
            labels_arr   = data['labels']
            unique_labels = np.unique(labels_arr)
            indices_list  = []
            for label in unique_labels:
                cluster_indices = np.where(labels_arr == label)[0]
                n_from = max(1, int(SUBSET_SIZE * len(cluster_indices) / total_cells))
                sampled = (np.random.choice(cluster_indices, size=n_from, replace=False)
                           if n_from < len(cluster_indices) else cluster_indices)
                indices_list.append(sampled)
            indices = np.sort(np.concatenate(indices_list))
        else:
            print("Falling back to random sampling...")
            indices = np.sort(np.random.choice(total_cells, size=SUBSET_SIZE, replace=False))

        expression        = expression[indices, :]
        data['cells']     = [data['cells'][i] for i in indices]
        data['metadata']  = data['metadata'].iloc[indices].copy()
        if data['adjacency'] is not None:
            data['adjacency'] = data['adjacency'][indices][:, indices]
        if data['labels'] is not None:
            data['labels'] = data['labels'][indices]
        if data['umap'] is not None:
            data['umap'] = data['umap'].iloc[indices].copy()

        print(f"\n SUBSET APPLIED: {expression.shape[0]} cells × {expression.shape[1]} genes")
    else:
        print("Using full dataset")

    total_cells = expression.shape[0]

# Keep a reference to the (subsetted) sparse expression for marker scoring later
expression_for_scoring = expression

# ============================================================================
# STEP 2: PCA + NORMALIZE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: PCA + Normalize")
print("=" * 80)

n_cells, n_genes = expression.shape
print(f"Working with: {n_cells} cells × {n_genes} genes")

if n_genes > N_PCS:
    with StepTimer("Step 2a: PCA"):
        print(f"\nApplying TruncatedSVD: {n_genes} genes → {N_PCS} PCs")
        svd  = TruncatedSVD(n_components=N_PCS, random_state=42)
        X_np = svd.fit_transform(expression)
        print(f"  Variance explained: {svd.explained_variance_ratio_.sum():.1%}")
        print(f"  PCA embedding shape: {X_np.shape}")
else:
    print("\nUsing raw expression (n_genes <= N_PCS)")
    X_np = expression.toarray() if issparse(expression) else expression

with StepTimer("Step 2c: Normalize"):
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_np   = scaler.fit_transform(X_np).astype(np.float32)
    print("  Features normalized (mean=0, std=1)")

# Build true_labels for training performance logging only
true_labels = None
if data['labels'] is not None:
    labels_arr    = data['labels']
    unique_labels = np.unique(labels_arr)
    n_clusters    = len(unique_labels)
    print(f"\nProcessing labels for training evaluator: {n_clusters} clusters")
    if n_clusters > 20:
        from sklearn.cluster import KMeans
        n_top = max(5, n_clusters // 4)
        print(f"  Creating 2-level hierarchy: {n_top} (top) / {n_clusters} (middle)")
        labels_top  = KMeans(n_clusters=n_top, random_state=42, n_init=10).fit_predict(X_np)
        true_labels = [labels_top, labels_arr]
    else:
        true_labels = [labels_arr]

# ============================================================================
# STEP 2d: BUILD GRAPH
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2d: Build Graph")
print("=" * 80)

with StepTimer("Step 2d: Build Graph"):
    print("\nBuilding KNN correlation graph from PCA features...")
    A_graph, A_np = get_input_graph(X=X_np, method='Correlation', K=K_NEIGHBORS, metric='1-R^2')

print(f"  Graph: {A_graph.number_of_nodes()} nodes, {A_graph.number_of_edges()} edges")
print(f"  Adjacency density: {A_np.sum() / (n_cells**2):.4f}")

A     = torch.clamp(torch.FloatTensor(A_np) + torch.eye(n_cells), 0, 1)
X     = torch.FloatTensor(X_np)
nodes, features = X.shape

# ============================================================================
# STEP 3: COMMUNITY SIZES — FORCE TOP = 2
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Community Sizes (forcing top-level = 2)")
print("=" * 80)

with StepTimer("Step 3: Community Size Estimation"):
    try:
        comm_middle, comm_top = compute_kappa(X, A, method='bethe_hessian', verbose=True)
        print(f"Bethe-Hessian estimated: Middle={comm_middle}, Top={comm_top}")
    except Exception as e:
        print(f"Bethe-Hessian failed ({e}), using heuristic")
        comm_middle = max(10, nodes // 50)

    comm_middle = max(4, comm_middle)   # guard against tiny subsets
    comm_sizes  = [comm_middle, 2]      # HCD reverses this → [2, comm_middle] internally
    print(f"comm_sizes passed to HCD: {comm_sizes}  (top forced to 2, middle={comm_middle})")

# ============================================================================
# STEP 4: CREATE MODEL
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

with StepTimer("Step 4: Model Creation"):
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
# STEP 5: TRAIN
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
reset_forward_timing()
with StepTimer("Step 5: Training"):
    output = trainer.fit(DEVICE)

_gate_encoder_s = forward_timing['gate_encoder']
_gate_decoder_s = forward_timing['gate_decoder']
_clustering_s   = forward_timing['clustering']
_forward_calls  = forward_timing['calls']

# ============================================================================
# STEP 6: EXTRACT CLUSTER ASSIGNMENTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Extracting Cluster Assignments")
print("=" * 80)

top_preds   = output.predicted_train['top'].cpu().numpy()   # (n_trained_cells,)
n_preds     = len(top_preds)
cell_ids    = data['cells'][:n_preds]
seurat_cl   = data['metadata']['seurat_clusters'].values[:n_preds]

unique_clusters = sorted(np.unique(top_preds))
print(f"Top-level cluster IDs found: {unique_clusters}  (expected 2)")
for cid in unique_clusters:
    print(f"  Cluster {cid}: {(top_preds == cid).sum()} cells")

# ============================================================================
# STEP 7: MARKER GENE SCORING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Marker Gene Scoring")
print("=" * 80)

gene_index = {g: i for i, g in enumerate(data['genes'])}

glial_present = [g for g in GLIAL_MARKERS      if g in gene_index]
epi_present   = [g for g in EPITHELIAL_MARKERS if g in gene_index]
missing_glial = [g for g in GLIAL_MARKERS      if g not in gene_index]
missing_epi   = [g for g in EPITHELIAL_MARKERS if g not in gene_index]

print(f"\nGlial markers found ({len(glial_present)}/{len(GLIAL_MARKERS)}): {glial_present}")
if missing_glial:
    print(f"  Missing: {missing_glial}")
print(f"Epithelial markers found ({len(epi_present)}/{len(EPITHELIAL_MARKERS)}): {epi_present}")
if missing_epi:
    print(f"  Missing: {missing_epi}")

if not glial_present:
    raise RuntimeError("No glial marker genes found in the dataset. Cannot annotate.")
if not epi_present:
    raise RuntimeError("No epithelial marker genes found in the dataset. Cannot annotate.")

# Extract marker expression from the raw sparse matrix (cells × genes, subsetted)
expr_sub  = expression_for_scoring[:n_preds, :]

glial_idx = [gene_index[g] for g in glial_present]
epi_idx   = [gene_index[g] for g in epi_present]
all_idx   = glial_idx + epi_idx
all_markers = glial_present + epi_present

# Slice only marker columns (small: n_cells × ~10) — avoid densifying the full matrix
glial_expr  = np.asarray(expr_sub[:, glial_idx].todense())   # (n_preds, n_glial)
epi_expr    = np.asarray(expr_sub[:, epi_idx].todense())     # (n_preds, n_epi)
marker_expr = np.asarray(expr_sub[:, all_idx].todense())     # (n_preds, n_all) for heatmap

# Per-cluster mean expression
cluster_scores = {}
marker_means   = np.zeros((len(unique_clusters), len(all_markers)))

print("\nPer-cluster marker scores:")
for i, cid in enumerate(unique_clusters):
    mask        = (top_preds == cid)
    glial_mean  = float(glial_expr[mask].mean())
    epi_mean    = float(epi_expr[mask].mean())
    cluster_scores[cid] = {'glial': glial_mean, 'epithelial': epi_mean}
    marker_means[i]     = marker_expr[mask].mean(axis=0)
    print(f"  Cluster {cid}: glial={glial_mean:.5f}, epithelial={epi_mean:.5f}")

# ============================================================================
# STEP 8: ANNOTATION — ASSIGN EPITHELIAL / GLIAL LABELS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Assigning Cell Type Labels")
print("=" * 80)

cluster_labels = {}

if len(unique_clusters) == 1:
    # Edge case: only one cluster (shouldn't happen with comm_sizes=[...,2])
    cid = unique_clusters[0]
    scores = cluster_scores[cid]
    cluster_labels[cid] = 'glial' if scores['glial'] >= scores['epithelial'] else 'epithelial'
    print(f"WARNING: Only 1 cluster found. Labelling as {cluster_labels[cid]}.")
else:
    # Rank clusters by (glial_score - epi_score) descending
    # → cluster with highest relative glial score = glial, other = epithelial
    ranked = sorted(unique_clusters,
                    key=lambda c: cluster_scores[c]['glial'] - cluster_scores[c]['epithelial'],
                    reverse=True)
    cluster_labels[ranked[0]]  = 'glial'
    cluster_labels[ranked[-1]] = 'epithelial'
    # Any extra clusters (>2, unlikely) get labelled 'other'
    for cid in ranked[1:-1]:
        cluster_labels[cid] = 'other'

for cid in unique_clusters:
    scores = cluster_scores[cid]
    print(f"  Cluster {cid} → {cluster_labels[cid]}"
          f"  (glial={scores['glial']:.5f}, epithelial={scores['epithelial']:.5f})")

deephcd_labels = np.array([cluster_labels[c] for c in top_preds])

# ============================================================================
# STEP 9: SEURAT COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Comparing with Seurat Clusters")
print("=" * 80)

results_df = pd.DataFrame({
    'cell':                cell_ids,
    'deephcd_top_cluster': top_preds.astype(int),
    'deephcd_label':       deephcd_labels,
    'seurat_cluster':      seurat_cl.astype(int),
})

crosstab = pd.crosstab(
    results_df['deephcd_label'],
    results_df['seurat_cluster'],
    margins=True
)
print("\nDeepHCD label vs Seurat cluster:")
print(crosstab.to_string())

# ============================================================================
# STEP 10: SAVE OUTPUTS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Saving Outputs")
print("=" * 80)

with StepTimer("Step 10: Save Outputs"):
    # Primary results CSV
    csv_path = os.path.join(OUTPUT_PATH, 'cell_type_assignments.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Assignments saved: {csv_path}")

    # Seurat crosstab
    crosstab_path = os.path.join(OUTPUT_PATH, 'deephcd_vs_seurat_crosstab.csv')
    crosstab.to_csv(crosstab_path)
    print(f"✓ Crosstab saved: {crosstab_path}")

    # Marker gene heatmap
    row_labels   = [f"Cluster {cid} ({cluster_labels[cid]})" for cid in unique_clusters]
    heatmap_data = pd.DataFrame(
        np.log1p(marker_means),
        index=row_labels,
        columns=all_markers
    )
    fig, ax = plt.subplots(figsize=(max(8, len(all_markers) * 0.9), max(3, len(unique_clusters) * 1.2)))
    sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', annot=True, fmt='.3f',
                linewidths=0.5, cbar_kws={'label': 'log1p(mean expression)'})
    ax.axvline(x=len(glial_present), color='navy', linewidth=2, linestyle='--')
    # Column group labels
    ax.text(len(glial_present) / 2, -0.5, 'Glial',
            ha='center', transform=ax.get_xaxis_transform(), fontsize=9, fontweight='bold')
    ax.text(len(glial_present) + len(epi_present) / 2, -0.5, 'Epithelial',
            ha='center', transform=ax.get_xaxis_transform(), fontsize=9, fontweight='bold')
    ax.set_title('Marker Gene Expression per DeepHCD Cluster')
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_PATH, 'marker_gene_heatmap.png')
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"✓ Marker heatmap saved: {heatmap_path}")

    # UMAP plot
    if data['umap'] is not None:
        umap_df  = data['umap']
        plot_df  = results_df.set_index('cell').join(umap_df, how='inner')
        n_lost   = len(results_df) - len(plot_df)
        if n_lost > 0:
            print(f"  WARNING: {n_lost} cells had no UMAP coordinates and were dropped from plot")

        color_map = {'glial': '#E74C3C', 'epithelial': '#3498DB', 'other': '#95A5A6'}
        fig, ax = plt.subplots(figsize=(8, 7))
        for label, grp in plot_df.groupby('deephcd_label'):
            ax.scatter(grp['UMAP_1'], grp['UMAP_2'],
                       c=color_map.get(label, '#95A5A6'),
                       label=label, s=8, alpha=0.7, rasterized=True)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('DeepHCD Cell Type Labels')
        ax.legend(markerscale=3, frameon=True)
        plt.tight_layout()
        umap_path = os.path.join(OUTPUT_PATH, 'umap_deephcd_labels.png')
        plt.savefig(umap_path, dpi=150)
        plt.close()
        print(f"✓ UMAP plot saved: {umap_path}")
    else:
        print("  UMAP plot skipped (no umap_embeddings.csv found)")

    # Save model
    model_path = os.path.join(OUTPUT_PATH, 'trained_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'comm_sizes': comm_sizes,
        'cluster_labels': cluster_labels,
        'subset_size': nodes,
    }, model_path)
    print(f"✓ Model saved: {model_path}")

# ============================================================================
# TIMING SUMMARY
# ============================================================================

tracemalloc.stop()
end_time = time.perf_counter()
elapsed  = end_time - start_time

_step_times["  ↳ GATE Encoder (total)"] = (_gate_encoder_s, None)
_step_times["  ↳ GATE Decoder (total)"] = (_gate_decoder_s, None)
_step_times["  ↳ Clustering (total)"]   = (_clustering_s,   None)
_step_times[f"  ↳ forward() calls"]     = (_forward_calls,  None)

time_file = os.path.join(OUTPUT_PATH, "execution_time.txt")
with open(time_file, "w") as _tf:
    _tf.write(f"Total execution time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
    _tf.write(f"Cells: {nodes}, Subset method: {SUBSET_METHOD}\n\n")
    _tf.write(f"{'Step':<45} {'Time':>10}  {'Peak MB':>10}\n")
    _tf.write("-" * 70 + "\n")
    for _name, (_val, _mb) in _step_times.items():
        if _name.startswith("  ↳ forward"):
            _tf.write(f"  {'↳ forward() calls':<43} {int(_val):>10}\n")
        elif _name.startswith("  ↳"):
            _tf.write(f"{_name:<45} {_fmt(_val):>10}\n")
        else:
            _mb_str = f"{_mb:.1f}" if _mb is not None else "  n/a"
            _tf.write(f"{_name:<45} {_fmt(_val):>10}  {_mb_str:>10}\n")
    _tf.write("-" * 70 + "\n")
    _tf.write(f"{'TOTAL':<45} {_fmt(elapsed):>10}\n")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Trained on {nodes} / {total_cells} cells")
print(f"✓ Top-level clusters: {len(unique_clusters)}")
for cid in unique_clusters:
    n = int((top_preds == cid).sum())
    scores = cluster_scores[cid]
    print(f"  Cluster {cid} ({cluster_labels[cid]}): {n} cells  "
          f"[glial={scores['glial']:.5f}, epi={scores['epithelial']:.5f}]")
print(f"\n{'Step':<45} {'Time':>10}  {'Peak MB':>10}")
print("-" * 70)
for _name, (_val, _mb) in _step_times.items():
    if _name.startswith("  ↳ forward"):
        print(f"  {'↳ forward() calls':<43} {int(_val):>10}")
    elif _name.startswith("  ↳"):
        print(f"{_name:<45} {_fmt(_val):>10}")
    else:
        _mb_str = f"{_mb:.1f}" if _mb is not None else "  n/a"
        print(f"{_name:<45} {_fmt(_val):>10}  {_mb_str:>10}")
print("-" * 70)
print(f"{'TOTAL':<45} {_fmt(elapsed):>10}")
print(f"\n✓ Results: {OUTPUT_PATH}")
print("=" * 80)

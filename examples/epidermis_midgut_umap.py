"""
epidermis_midgut_umap.py
------------------------
Generate a UMAP of the epidermis/midgut 2000-cell subset used in
epidermis_midgut_training.py.

Steps:
  1. Load data.h5ad
  2. Filter to epidermis + midgut cells
  3. Stratified subsample to 2000 cells (same seed as training script)
  4. UMAP to 2D (directly on raw expression)
  5. Save plots coloured by cell type and (if available) Seurat cluster

Output saved to: <repo_root>/epidermis_midgut_umap/

Launch:
    python -u epidermis_midgut_umap.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import umap

# Repo root is three levels up from this script (DeepHCD_copy/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from load_h5ad import load_h5ad_data, filter_cell_types

# ── Configuration ─────────────────────────────────────────────────────────────

TARGET_CELL_TYPES = ['epidermis', 'midgut']
SUBSET_SIZE  = 2000
RANDOM_SEED  = 42

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H5AD_PATH = os.environ.get(
    'DEEPHCD_DATA_DIR',
    os.path.join(_REPO_ROOT, 'converted_data_h5ad', 'data.h5ad')
)
OUTPUT_PATH = os.environ.get(
    'DEEPHCD_UMAP_DIR',
    os.path.join(_REPO_ROOT, 'epidermis_midgut_umap')
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 70)
print("UMAP  —  Epidermis vs Midgut  (2000-cell subset)")
print("=" * 70)
print(f"H5AD  : {H5AD_PATH}")
print(f"Output: {OUTPUT_PATH}\n")

# ── Step 1: Load ──────────────────────────────────────────────────────────────

print("Step 1: Loading data …")
data = load_h5ad_data(H5AD_PATH)

# ── Step 2: Filter to target cell types ──────────────────────────────────────

print("\nStep 2: Filtering to", TARGET_CELL_TYPES, "…")
data = filter_cell_types(data, TARGET_CELL_TYPES)

expression    = data['expression']
meta          = data['metadata']
seurat_labels = data['labels']

sorted_types = sorted(TARGET_CELL_TYPES)
ct_map = {name: i for i, name in enumerate(sorted_types)}
ct_labels = np.array([ct_map[v] for v in meta['manual_annot'].values])

print(f"  Cell-type encoding: {ct_map}")
for name, idx in ct_map.items():
    print(f"    {idx} = {name}  ({(ct_labels == idx).sum():,} cells)")

# ── Step 3: Stratified subsample to 2000 cells ───────────────────────────────

print(f"\nStep 3: Stratified subsample → {SUBSET_SIZE} cells …")
total = expression.shape[0]
np.random.seed(RANDOM_SEED)

sub_indices_list = []
for name, label_id in ct_map.items():
    ct_idx = np.where(ct_labels == label_id)[0]
    n_from = max(1, int(SUBSET_SIZE * len(ct_idx) / total))
    sampled = (np.random.choice(ct_idx, size=n_from, replace=False)
               if n_from < len(ct_idx) else ct_idx)
    sub_indices_list.append(sampled)
    print(f"  {name}: {len(sampled):,} / {len(ct_idx):,}")

sub_indices = np.sort(np.concatenate(sub_indices_list))
expression  = expression[sub_indices, :]
meta        = meta.iloc[sub_indices].reset_index(drop=True)
ct_labels   = ct_labels[sub_indices]
if seurat_labels is not None:
    seurat_labels = seurat_labels[sub_indices]

print(f"  Final: {expression.shape[0]:,} cells × {expression.shape[1]:,} genes")

# Dense conversion (2000 cells is manageable in memory)
X = (expression.toarray() if hasattr(expression, 'toarray') else np.array(expression)).astype(np.float32)

# ── Step 4: UMAP ──────────────────────────────────────────────────────────────

print("\nStep 4: Running UMAP …")
reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=30,
                    min_dist=0.3, metric='euclidean')
embedding = reducer.fit_transform(X)
print(f"  Embedding shape: {embedding.shape}")

# ── Step 5: Plot ──────────────────────────────────────────────────────────────

def scatter(ax, embedding, labels, title, cmap='Set1'):
    unique = sorted(set(labels.tolist() if hasattr(labels, 'tolist') else labels))
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(unique), 1)))
    for color, label in zip(colors, unique):
        mask = np.array(labels) == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[color], label=str(label),
                   s=8, alpha=0.6, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(markerscale=2, fontsize=8, loc='best', framealpha=0.7)

print("\nStep 5: Saving plots …")

# Plot 1 — coloured by true cell type
true_labels_str = np.array([sorted_types[v] for v in ct_labels])
fig, ax = plt.subplots(figsize=(7, 6))
scatter(ax, embedding, true_labels_str, 'UMAP — True Cell Type', cmap='Set1')
plt.tight_layout()
p = os.path.join(OUTPUT_PATH, 'umap_cell_type.png')
plt.savefig(p, dpi=150)
plt.close()
print(f"  Saved: {p}")

# Plot 2 — coloured by Seurat cluster (if available)
if seurat_labels is not None:
    n_clusters = len(np.unique(seurat_labels))
    cmap = 'tab20' if n_clusters > 10 else 'tab10'
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter(ax, embedding, seurat_labels.astype(str),
            f'UMAP — Seurat Cluster ({n_clusters} clusters)', cmap=cmap)
    plt.tight_layout()
    p = os.path.join(OUTPUT_PATH, 'umap_seurat_cluster.png')
    plt.savefig(p, dpi=150)
    plt.close()
    print(f"  Saved: {p}")

# Plot 3 — side-by-side
ncols = 2 if seurat_labels is not None else 1
fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
if ncols == 1:
    axes = [axes]
scatter(axes[0], embedding, true_labels_str, 'True Cell Type', cmap='Set1')
if seurat_labels is not None:
    scatter(axes[1], embedding, seurat_labels.astype(str), 'Seurat Cluster', cmap='tab20')
plt.suptitle('UMAP: Epidermis vs Midgut (2 000-cell subset)', fontsize=13)
plt.tight_layout()
p = os.path.join(OUTPUT_PATH, 'umap_comparison.png')
plt.savefig(p, dpi=150)
plt.close()
print(f"  Saved: {p}")

print("\nDone. All plots saved to:", OUTPUT_PATH)

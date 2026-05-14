import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import pandas as pd
import loompy
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
NEON_PALETTE = [
    '#FF00FF', '#00FFFF', '#39FF14', '#FFFF00', '#FF6EC7',
    '#FF5F1F', '#BC13FE', '#1F51FF', '#CCFF00', '#FF073A',
    '#00FF9F', '#FE4164', '#04D9FF', '#F4FF61', '#FF9933',
    '#8AFF00', '#FA00FF', '#00FFEF', '#FFB200', '#D5FF00',
]
NEON_CMAP = ListedColormap(NEON_PALETTE)
import numpy as np
import umap
import torch
import anndata
from deephcd.utils.utilities import get_input_graph
from deephcd.utils.utilities import LoadData, compute_kappa
from deephcd.model.model import HCD
from deephcd.model.train import Trainer


RUN_UMAP_ORIGINAL = False
TSNE = False
SUBSET = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_PATH = './retina_training_results/'

#Reading in retina.loom data
with loompy.connect('/Users/jordandavis/Documents/DeepHCD_copy/retina.loom') as ds:
    X = ds[:, :].T  # loompy is genes×cells, AnnData wants cells×genes
    obs = pd.DataFrame({k: v.squeeze() for k, v in ds.col_attrs.items()})
    var = pd.DataFrame({k: v.squeeze() for k, v in ds.row_attrs.items()},
                       index=ds.ra.get('Gene', ds.ra.get('Accession', np.arange(ds.shape[0]))))
    cell_ids = obs['CellID'].astype(str) if 'CellID' in obs.columns else pd.RangeIndex(X.shape[0]).astype(str)
    obs.index = pd.Index(cell_ids, dtype=str)
    retina_data = anndata.AnnData(X=X, obs=obs, var=var)
print(retina_data)
print("\n--- Cell metadata (obs) ---")
print(retina_data.obs.head())
print("\n--- Gene metadata (var) ---")
print(retina_data.var.head())
print("\n--- Expression matrix (X) ---")
print(pd.DataFrame(retina_data.X[:10, :10],
                   index=retina_data.obs.index[:10],
                   columns=retina_data.var.index[:10]))

X_dense = retina_data.X.astype(np.float32)

# PCA 
print("\nRunning PCA (50 components)...")
pca = TruncatedSVD(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_dense)

if TSNE:
    n_cells = X_pca.shape[0]
    print(f"\nRunning t-SNE on {n_cells} cells...")
    tsne = TSNE(n_components=2, random_state=42,perplexity=min(30, n_cells - 1), max_iter=1000)
    tsne_embedding = tsne.fit_transform(X_pca)
    cluster_labels = retina_data.obs['ClusterID'].astype(int).values
    batch_labels   = retina_data.obs['BatchID'].astype(int).values
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc0 = axes[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=cluster_labels, cmap=NEON_CMAP, s=5, alpha=0.6)
    sc1 = axes[1].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=batch_labels,   cmap=NEON_CMAP, s=5, alpha=0.6)
    plt.colorbar(sc0, ax=axes[0])
    plt.colorbar(sc1, ax=axes[1])
    axes[0].set_title('t-SNE - Cluster ID')
    axes[1].set_title('t-SNE - Batch ID')
    plt.tight_layout()

    output_path = '/Users/jordandavis/Documents/DeepHCD_copy/retina_tsne.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    
# ── UMAP ──────────────────────────────────────────────────────────────────────
if RUN_UMAP_ORIGINAL:
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    embedding = reducer.fit_transform(X_pca)

    cluster_labels = retina_data.obs['ClusterID'].astype(int).values
    batch_labels   = retina_data.obs['BatchID'].astype(int).values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc0 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap=NEON_CMAP, s=5, alpha=0.6)
    sc1 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=batch_labels,   cmap=NEON_CMAP, s=5, alpha=0.6)
    plt.colorbar(sc0, ax=axes[0])
    plt.colorbar(sc1, ax=axes[1])
    axes[0].set_title('UMAP — Cluster ID')
    axes[1].set_title('UMAP — Batch ID')
    plt.tight_layout()

    output_path = '/Users/jordandavis/Documents/DeepHCD_copy/retina_umap.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

#subsetting
if SUBSET != None:
    np.random.seed(42)
    idx = np.random.choice(X_pca.shape[0],size=SUBSET,replace=False)
    X = X_pca[idx,:]
    obs = retina_data.obs.iloc[idx]
    print(X.shape)


#Building graph
print('Building Adjacency', flush=True)
A_graph, A_np = get_input_graph(X=X, method='Correlation', K=30, metric='1-R^2', r_cutoff=0.7)
print(A_np.shape, flush=True)

print('Converting to tensors', flush=True)
X = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32)
print('X converted', flush=True)
A = torch.tensor(np.ascontiguousarray(A_np), dtype=torch.float32) + torch.eye(A_np.shape[0])
print('A converted', flush=True)
nodes, features = X.shape

#Estimate community size
print('Estimating community sizes', flush=True)
A_np = torch.from_numpy(A_np)
print('A_np converted', flush=True)
comm_sizes = compute_kappa(X, A, method='bethe_hessian', verbose=False)
print(f'Predicted Community Sizes: {comm_sizes}')

#Model Building
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

#Train model
print("\n4. Training model...")
trainer = Trainer(
    model=model,
    X=X,
    A=A,
    epochs=30,                      # Number of training epochs
    learning_rate=1e-3,             # Learning rate
    batch_size=32,                  # Batch size
    early_stopping=True,            # Enable early stopping
    patience=5,                     # Stop if no improvement for 5 epochs
    output_path=OUTPUT_PATH,           # Output directory
    verbose=True
)
print('done training')
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
    if len(final_perf) > 0:
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

# t-SNE comparison: actual ClusterID vs predicted top/middle labels (on the subset used for training)
from sklearn.manifold import TSNE as SKTSNE
print("\nRunning t-SNE on the trained subset for label comparison...")
X_sub_pca = X_pca[idx, :]
tsne_sub = SKTSNE(n_components=2, random_state=42,
                  perplexity=min(30, X_sub_pca.shape[0] - 1),
                  max_iter=1000).fit_transform(X_sub_pca)

actual_labels = obs['ClusterID'].astype(int).values
top_pred    = output.predicted_train['top'].detach().cpu().numpy()
middle_pred = output.predicted_train['middle'].detach().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, labs, title in zip(
    axes,
    [actual_labels, top_pred, middle_pred],
    ['Actual ClusterID', 'Predicted Top Layer', 'Predicted Middle Layer'],
):
    sc = ax.scatter(tsne_sub[:, 0], tsne_sub[:, 1], c=labs, cmap=NEON_CMAP, s=6, alpha=0.7)
    plt.colorbar(sc, ax=ax)
    ax.set_title(title)
plt.tight_layout()
compare_path = '/Users/jordandavis/Documents/DeepHCD_copy/retina_tsne_predicted_vs_actual.png'
plt.savefig(compare_path, dpi=150)
plt.close()
print(f"Saved: {compare_path}")
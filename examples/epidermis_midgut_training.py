"""
epidermis_midgut_training.py
----------------------------
Train DeepHCD on cells annotated as 'epidermis' or 'midgut' only.

Reads from converted_data_joined/ (produced by join_cell_types.py), filters to
the two target cell types, then runs the standard DeepHCD pipeline.

true_labels uses cell type (0=epidermis, 1=midgut) so training performance
metrics reflect how well the model separates these two populations.

Launch (single process):
    python -u epidermis_midgut_training.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import time
import tracemalloc

# h5ad loader — lives two levels up from this script (DeepHCD_copy/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from load_h5ad import load_h5ad_data, filter_cell_types

from deephcd.model.model import HCD
try:
    from deephcd.model.model import forward_timing, reset_forward_timing
except ImportError:
    forward_timing = {'gate_encoder': 0.0, 'gate_decoder': 0.0, 'clustering': 0.0, 'calls': 0}
    def reset_forward_timing():
        pass
from deephcd.model.train import Trainer
from deephcd.utils.utilities import compute_kappa
from deephcd.utils.utilities import get_input_graph
from deephcd.utils.train_utils import split_dataset

# ── Distributed setup (unchanged from single_cell_training.py) ────────────────

def _init_distributed():
    if not dist.is_available():
        return 0, 1
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)
    elif 'SLURM_PROCID' in os.environ:
        rank       = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK']       = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        # Only set MASTER_PORT/ADDR if not already provided by the SLURM script.
        # SLURMD_NODENAME is the *current* node and differs per rank — using it
        # as MASTER_ADDR causes every rank to think it is the master, breaking
        # the Gloo TCP rendezvous with a bad_alloc crash.
        if 'MASTER_PORT' not in os.environ:
            job_id = int(os.environ.get('SLURM_JOB_ID', 0))
            os.environ['MASTER_PORT'] = str(29500 + job_id % 10000)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        dist.init_process_group(backend='mpi')
    else:
        return 0, 1
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size

RANK, WORLD_SIZE = _init_distributed()
IS_MAIN = RANK == 0

# When rank 0 crashes with an uncaught exception before the first broadcast,
# other ranks hang waiting for it. This hook fires on rank 0 before Python
# exits, broadcasts a failure flag so all ranks can shut down cleanly.
_ok = [True]
if WORLD_SIZE > 1:
    _orig_excepthook = sys.excepthook
    def _excepthook(exc_type, exc_val, exc_tb):
        if IS_MAIN:
            try:
                dist.broadcast_object_list([False], src=0)
            except Exception:
                pass
        _orig_excepthook(exc_type, exc_val, exc_tb)
    sys.excepthook = _excepthook

def log(*args, **kwargs):
    if IS_MAIN:
        print(*args, **kwargs, flush=True)

# ── Timing ────────────────────────────────────────────────────────────────────

_step_times = {}

class StepTimer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        if IS_MAIN:
            tracemalloc.clear_traces()
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t0
        if IS_MAIN:
            _, peak = tracemalloc.get_traced_memory()
            _step_times[self.name] = (elapsed, peak / 1024 / 1024)
            print(f"  [{self.name}] done in {_fmt(elapsed)}", flush=True)

def _fmt(seconds):
    return f"{seconds/60:.1f} min" if seconds >= 60 else f"{seconds:.2f}s"

if IS_MAIN:
    tracemalloc.start()
start_time = time.perf_counter()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Cell types to keep — must match manual_annot strings in rna_meta.rds
TARGET_CELL_TYPES = ['epidermis', 'midgut']

# Subsetting: use stratified sampling so both cell types are represented
USE_SUBSET    = True
SUBSET_SIZE   = 2000    # total cells after filter; set to None to use all
SUBSET_METHOD = 'stratified'
RANDOM_SEED   = 42

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H5AD_PATH = os.environ.get(
    'DEEPHCD_DATA_DIR',
    os.path.join(_SCRIPT_DIR, 'converted_data_h5ad', 'data.h5ad')
)
OUTPUT_PATH = os.environ.get(
    'DEEPHCD_OUTPUT_DIR',
    os.path.join(_SCRIPT_DIR, 'epidermis_midgut_test_nonzero_loss')
)
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
N_PCS   = 100
K_NEIGHBORS = 30

if IS_MAIN:
    os.makedirs(OUTPUT_PATH, exist_ok=True)

log("=" * 80)
log("DeepHCD  —  Epidermis vs Midgut")
log("=" * 80)
log(f"\nTarget cell types : {TARGET_CELL_TYPES}")
log(f"Device            : {DEVICE}")
log(f"H5AD file         : {H5AD_PATH}")
log(f"Output directory  : {OUTPUT_PATH}")

# ============================================================================
# STEP 1: Load data from h5ad
# ============================================================================

log("\n" + "=" * 80)
log("STEP 1: Loading Data")
log("=" * 80)

if IS_MAIN:
    with StepTimer("Step 1: Load Data"):
        data = load_h5ad_data(H5AD_PATH)

# ============================================================================
# STEP 1.5: Filter to target cell types
# ============================================================================

    log("\n" + "=" * 80)
    log("STEP 1.5: Filtering to target cell types")
    log("=" * 80)

    with StepTimer("Step 1.5: Filter"):
        total_before = len(data['cells'])
        log(f"Total cells before filter : {total_before:,}")

        data = filter_cell_types(data, TARGET_CELL_TYPES)

        expression     = data['expression']
        cells_filtered = data['cells']
        meta_filtered  = data['metadata']
        adj_filtered   = data['adjacency']

        # Encode target cell types as binary true labels (alphabetical order)
        sorted_types = sorted(TARGET_CELL_TYPES)
        ct_map = {name: i for i, name in enumerate(sorted_types)}
        ct_labels_filtered = np.array(
            [ct_map[v] for v in meta_filtered['manual_annot'].values]
        )
        log(f"\nCell-type label encoding: {ct_map}")
        for name, idx in ct_map.items():
            log(f"  {idx} = {name}  ({(ct_labels_filtered == idx).sum():,} cells)")

        # Seurat cluster labels for the filtered subset — used as fine labels
        seurat_labels_filtered = data['labels']
        if seurat_labels_filtered is not None:
            log(f"\nSeurat clusters in subset: {len(np.unique(seurat_labels_filtered))}")

    # ── Optional stratified subsetting ──────────────────────────────────────
    log("\n" + "=" * 80)
    log("STEP 1.6: Subsetting")
    log("=" * 80)

    with StepTimer("Step 1.6: Subset"):
        total_filtered = expression.shape[0]

        if USE_SUBSET and SUBSET_SIZE and total_filtered > SUBSET_SIZE:
            np.random.seed(RANDOM_SEED)
            log(f"Stratified subsetting: {SUBSET_SIZE} / {total_filtered} cells")
            sub_indices_list = []
            for name, label_id in ct_map.items():
                ct_idx = np.where(ct_labels_filtered == label_id)[0]
                n_from = max(1, int(SUBSET_SIZE * len(ct_idx) / total_filtered))
                sampled = (np.random.choice(ct_idx, size=n_from, replace=False)
                           if n_from < len(ct_idx) else ct_idx)
                sub_indices_list.append(sampled)
                log(f"  {name}: {len(sampled):,} / {len(ct_idx):,}")
            sub_indices = np.sort(np.concatenate(sub_indices_list))

            expression             = expression[sub_indices, :]
            cells_filtered         = [cells_filtered[i] for i in sub_indices]
            meta_filtered          = meta_filtered.iloc[sub_indices].copy()
            ct_labels_filtered     = ct_labels_filtered[sub_indices]
            if adj_filtered is not None:
                adj_filtered       = adj_filtered[sub_indices][:, sub_indices]
            if seurat_labels_filtered is not None:
                seurat_labels_filtered = seurat_labels_filtered[sub_indices]
        else:
            log(f"Using all {total_filtered:,} filtered cells (no subsetting)")

        total_cells = expression.shape[0]
        log(f"\nFinal dataset: {total_cells:,} cells × {expression.shape[1]:,} genes")

# ============================================================================
# STEP 2: PCA + Normalize
# ============================================================================

    log("\n" + "=" * 80)
    log("STEP 2: PCA + Normalize")
    log("=" * 80)

    n_cells_local, n_genes = expression.shape

    if n_genes > N_PCS:
        with StepTimer("Step 2a: PCA"):
            log(f"TruncatedSVD: {n_genes} genes → {N_PCS} PCs")
            svd = TruncatedSVD(n_components=N_PCS, random_state=42)
            X_np = svd.fit_transform(expression)
            log(f"  Variance explained: {svd.explained_variance_ratio_.sum():.1%}")
    else:
        X_np = expression.toarray() if issparse(expression) else expression

    with StepTimer("Step 2c: Normalize"):
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np).astype(np.float32)

    # Build true_labels:
    #   [ct_labels]               if no Seurat clusters, or
    #   [ct_labels, seurat_labels] for a 2-level hierarchy
    true_labels = [ct_labels_filtered]
    if seurat_labels_filtered is not None:
        n_seurat = len(np.unique(seurat_labels_filtered))
        log(f"\nUsing 2-level true labels: "
            f"{len(ct_map)} cell types (top) + {n_seurat} Seurat clusters (middle)")
        true_labels = [ct_labels_filtered, seurat_labels_filtered]
    else:
        log(f"\nUsing 1-level true labels: {len(ct_map)} cell types")

    _broadcast_meta = [X_np.shape[0], X_np.shape[1], total_cells, true_labels]
else:
    X_np = None
    _broadcast_meta = [None, None, None, None]
    total_cells = 0

# Broadcast success/failure so non-main ranks can exit cleanly if rank 0 failed
if WORLD_SIZE > 1:
    dist.broadcast_object_list(_ok, src=0)
if not _ok[0]:
    log("Rank 0 encountered an error — all ranks exiting.")
    if WORLD_SIZE > 1:
        dist.destroy_process_group()
    sys.exit(1)

# Broadcast to other ranks if distributed
if WORLD_SIZE > 1:
    dist.broadcast_object_list(_broadcast_meta, src=0)
    n_cells_bc, n_features_bc, total_cells, true_labels = _broadcast_meta
    if not IS_MAIN:
        X_np = np.empty((n_cells_bc, n_features_bc), dtype=np.float32)
    X_t = torch.from_numpy(X_np)
    dist.broadcast(X_t, src=0)
    if not IS_MAIN:
        X_np = X_t.numpy()

log(f"\nAll ranks have X: {X_np.shape}")

# ============================================================================
# STEP 2d: Build Graph
# ============================================================================

log("\n" + "=" * 80)
log("STEP 2d: Build Graph")
log("=" * 80)

with StepTimer(f"Step 2d: Build Graph [rank {RANK}]"):
    if IS_MAIN and adj_filtered is not None:
        log("Using Seurat SNN submatrix for filtered cells...")
        A_np = (adj_filtered.toarray() if issparse(adj_filtered)
                else adj_filtered).astype(np.float32)
        log(f"  {A_np.shape[0]} nodes, {int((A_np > 0).sum())} edges")
    else:
        log("Building KNN graph from PCA features...")
        A_graph, A_np = get_input_graph(X=X_np, method='Correlation',
                                        K=K_NEIGHBORS, metric='1-R^2')
        log(f"  {A_graph.number_of_nodes()} nodes, {A_graph.number_of_edges()} edges")

A = torch.clamp(torch.FloatTensor(A_np) + torch.eye(X_np.shape[0]), 0, 1)
X = torch.FloatTensor(X_np)
nodes, features = X.shape
log(f"\nTensors: X={X.shape}, A={A.shape}")

# ============================================================================
# STEP 3: Estimate Community Sizes
# ============================================================================

log("\n" + "=" * 80)
log("STEP 3: Estimating Community Sizes")
log("=" * 80)

_comm_list = [None]
if IS_MAIN:
    with StepTimer("Step 3: Community Size Estimation"):
        try:
            comm_middle, comm_top = compute_kappa(X, A, method='bethe_hessian', verbose=True)
            # Floor at 2 since we have 2 cell types
            comm_top    = max(2, comm_top)
            comm_middle = max(comm_top, comm_middle)
            comm_sizes  = [comm_middle, comm_top]
            log(f"Bethe-Hessian: Top={comm_top}, Middle={comm_middle}")
        except Exception as e:
            log(f"Bethe-Hessian failed: {e}")
            comm_top    = 2
            comm_middle = max(4, nodes // 200)
            comm_sizes  = [comm_middle, comm_top]
            log(f"Using heuristic: Top={comm_top}, Middle={comm_middle}")
    _comm_list = [comm_sizes]

if WORLD_SIZE > 1:
    dist.broadcast_object_list(_comm_list, src=0)
comm_sizes = _comm_list[0]

# ============================================================================
# STEP 4: Create Model
# ============================================================================

log("\n" + "=" * 80)
log("STEP 4: Creating Model")
log("=" * 80)

if nodes < 1000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-3, 64, 200
    log(f"Small dataset: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
elif nodes < 5000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-4, 128, 150
    log(f"Medium dataset: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
elif nodes < 10000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-5, 64, 100
    log(f"Large dataset: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
else:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-5, 512, 100
    log(f"Very large dataset: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")

EARLY_STOPPING = True
PATIENCE = 10
PLOT_HEATMAPS = False
PLOT_TSNE     = True

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

log(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
log(f"Community sizes: {comm_sizes}  (top={comm_sizes[-1]}, middle={comm_sizes[0]})")

# ============================================================================
# STEP 5: Train
# ============================================================================

log("\n" + "=" * 80)
log("STEP 5: Training")
log("=" * 80)

if IS_MAIN:
    log("Splitting 80/20 train/validation...")
    train_set, val_set = split_dataset(X, A, labels=true_labels, split=[0.8, 0.2])
    X_train, A_train, labels_train = train_set
    log(f"  Train: {X_train.shape[0]:,} nodes | Val: {val_set[0].shape[0]:,} nodes")

    trainer = Trainer(
        model=model,
        X=X_train,
        A=A_train,
        validation_data=val_set,
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
        true_labels=labels_train,
        output_path=OUTPUT_PATH,
        save_output=True,
        use_logging=True,
        log_to_file=True,
        verbose=True
    )

    log("Starting training...\n")
    reset_forward_timing()
    with StepTimer("Step 5: Training"):
        output = trainer.fit(DEVICE)

    _total_forward_s   = forward_timing['total_forward']
    _input_norm_s      = forward_timing['input_norm']
    _edge_index_s      = forward_timing['edge_index']
    _gate_encoder_s    = forward_timing['gate_encoder']
    _dot_product_s     = forward_timing['dot_product']
    _gate_decoder_s    = forward_timing['gate_decoder']
    _output_layers_s   = forward_timing['output_layers']
    _top_comm_s        = forward_timing['top_comm']
    _select_subsets_s  = forward_timing['select_subsets']
    _middle_comm_s     = forward_timing['middle_comm']
    _clustering_s      = forward_timing['clustering']
    _batch_prep_s      = forward_timing['batch_prep']
    _loss_compute_s    = forward_timing['loss_compute']
    _backward_s        = forward_timing['backward']
    _grad_clip_s       = forward_timing['grad_clip']
    _optimizer_step_s  = forward_timing['optimizer_step']
    _gpu_cleanup_s     = forward_timing['gpu_cleanup']
    _validation_s      = forward_timing['validation']
    _perf_eval_s       = forward_timing['perf_eval']
    _forward_calls     = forward_timing['calls']

# ============================================================================
# STEP 6: Save Results
# ============================================================================

if IS_MAIN:
    log("\n" + "=" * 80)
    log("TRAINING COMPLETE!")
    log("=" * 80)

    if output.train_loss_history:
        log(f"\nFinal training loss: {output.train_loss_history[-1]['Total Loss']:.4f}")

    if output.performance_history:
        final_perf = next((p for p in reversed(output.performance_history) if p is not None), None)
        if final_perf:
            log("\nPerformance Metrics (against cell-type true labels):")
            layer_names = ['Cell type (top)', 'Seurat clusters (middle)']
            for i, perf in enumerate(final_perf):
                if perf is not None:
                    name = layer_names[i] if i < len(layer_names) else f'Layer {i}'
                    log(f"  {name}: H={perf[0]:.3f}, C={perf[1]:.3f}, "
                        f"NMI={perf[2]:.3f}, ARI={perf[3]:.3f}")

    with StepTimer("Step 6: Save Results"):
        save_model = model.module if isinstance(model, DDP) else model
        MODEL_PATH = os.path.join(OUTPUT_PATH, 'trained_model.pth')
        torch.save({
            'model_state_dict': save_model.state_dict(),
            'comm_sizes': comm_sizes,
            'subset_size': nodes,
            'cell_types': TARGET_CELL_TYPES,
            'ct_label_map': ct_map,
            'config': {
                'method': 'top_down',
                'ae_hidden_dims': [128, 64, 32],
                'comm_operator': 'Linear',
                'dropout': 0.3
            }
        }, MODEL_PATH)
        log(f"\nModel saved: {MODEL_PATH}")

        if hasattr(output, 'predicted_train'):
            top_preds = output.predicted_train.get('top')
            mid_preds = output.predicted_train.get('middle')
            top_arr = top_preds.cpu().numpy() if top_preds is not None else None
            mid_arr = mid_preds.cpu().numpy() if mid_preds is not None else None
            n_preds = (len(top_arr) if top_arr is not None else
                       len(mid_arr) if mid_arr is not None else len(cells_filtered))
            pred_df = pd.DataFrame({
                'cell':           cells_filtered[:n_preds],
                'true_cell_type': [sorted_types[v] for v in ct_labels_filtered[:n_preds]],
                'top_cluster':    top_arr if top_arr is not None else [None] * n_preds,
                'middle_cluster': mid_arr if mid_arr is not None else [None] * n_preds,
            })
            pred_path = os.path.join(OUTPUT_PATH, 'predictions.csv')
            pred_df.to_csv(pred_path, index=False)
            log(f"Predictions saved: {pred_path}")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    if not PLOT_HEATMAPS:
        log("Heatmap generation skipped (PLOT_HEATMAPS=False)")
    if PLOT_HEATMAPS:
        with StepTimer("Step 7: Heatmaps"):
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns

                top_preds = output.predicted_train.get('top')
                mid_preds = output.predicted_train.get('middle')
                top_arr = top_preds.cpu().numpy() if top_preds is not None else None
                mid_arr = mid_preds.cpu().numpy() if mid_preds is not None else None
                _n = (len(top_arr) if top_arr is not None else
                      len(mid_arr) if mid_arr is not None else len(cells_filtered))
                pred_df = pd.DataFrame({
                    'cell':           cells_filtered[:_n],
                    'true_cell_type': [sorted_types[v] for v in ct_labels_filtered[:_n]],
                    'top_cluster':    top_arr if top_arr is not None else [None] * _n,
                    'middle_cluster': mid_arr if mid_arr is not None else [None] * _n,
                })

                X_np_plot = X.numpy()[:_n]

                # Sort by true cell type then predicted top cluster
                sorted_df  = pred_df.dropna(subset=['top_cluster']).sort_values(
                    ['true_cell_type', 'top_cluster'])
                sorted_idx = sorted_df.index.to_numpy()
                corr_matrix = np.corrcoef(X_np_plot[sorted_idx])

                fig, ax = plt.subplots(figsize=(10, 9))
                sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', center=0,
                            vmin=-1, vmax=1, xticklabels=False, yticklabels=False,
                            cbar_kws={'label': 'Pearson Correlation'})
                # Draw dividers at cell-type boundaries
                ct_vals = sorted_df['true_cell_type'].values
                for b in np.where(np.diff(ct_vals != ct_vals[0]))[0] + 1:
                    ax.axhline(b, color='black', linewidth=1.5, linestyle='--')
                    ax.axvline(b, color='black', linewidth=1.5, linestyle='--')
                ax.set_title('Cell-Cell Correlation (sorted by cell type then cluster)')
                ax.set_xlabel('Cells')
                ax.set_ylabel('Cells')
                plt.tight_layout()
                heatmap1_path = os.path.join(OUTPUT_PATH, 'heatmap_cell_correlation.png')
                plt.savefig(heatmap1_path, dpi=150)
                plt.close()
                log(f"Cell-cell correlation heatmap saved: {heatmap1_path}")

                for level, col in [('top', 'top_cluster'), ('middle', 'middle_cluster')]:
                    co = pred_df.dropna(subset=[col])
                    if len(co) == 0:
                        continue
                    cluster_ids = sorted(co[col].astype(int).unique())
                    cluster_means = np.stack([
                        X_np_plot[co[co[col].astype(int) == c].index].mean(axis=0)
                        for c in cluster_ids
                    ])
                    inter_corr = np.corrcoef(cluster_means)
                    fig, ax = plt.subplots(figsize=(max(6, len(cluster_ids)),
                                                    max(5, len(cluster_ids) - 1)))
                    sns.heatmap(inter_corr, ax=ax, cmap='coolwarm', center=0,
                                vmin=-1, vmax=1, annot=True, fmt='.2f',
                                xticklabels=[f'C{c}' for c in cluster_ids],
                                yticklabels=[f'C{c}' for c in cluster_ids],
                                cbar_kws={'label': 'Pearson Correlation'})
                    ax.set_title(f'Inter-Cluster Correlation ({level.capitalize()} level)')
                    plt.tight_layout()
                    hm_path = os.path.join(OUTPUT_PATH, f'heatmap_cluster_corr_{level}.png')
                    plt.savefig(hm_path, dpi=150)
                    plt.close()
                    log(f"Cluster correlation heatmap saved: {hm_path}")

            except Exception as e:
                log(f"Heatmap generation failed: {e}")

    # ── t-SNE plots ───────────────────────────────────────────────────────────
    if not PLOT_TSNE:
        log("t-SNE generation skipped (PLOT_TSNE=False)")
    if PLOT_TSNE:
        with StepTimer("Step 8: t-SNE"):
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from sklearn.manifold import TSNE

                top_preds = output.predicted_train.get('top')
                mid_preds = output.predicted_train.get('middle')
                top_arr = top_preds.cpu().numpy() if top_preds is not None else None
                mid_arr = mid_preds.cpu().numpy() if mid_preds is not None else None
                _n = (len(top_arr) if top_arr is not None else
                      len(mid_arr) if mid_arr is not None else len(cells_filtered))

                X_np_plot = X.numpy()[:_n]

                log("  Running t-SNE (this may take a moment)...")
                tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                            perplexity=min(30, _n - 1), max_iter=1000)
                embedding = tsne.fit_transform(X_np_plot)

                true_labels_plot = np.array(
                    [sorted_types[v] for v in ct_labels_filtered[:_n]]
                )

                # Neon palette — maximally distinct, high-saturation colors
                NEON_PALETTE = [
                    '#FF00FF',  # magenta
                    '#00FFFF',  # cyan
                    '#39FF14',  # neon green
                    '#FFFF00',  # neon yellow
                    '#FF6EC7',  # neon pink
                    '#FF5F1F',  # neon orange
                    '#BC13FE',  # neon purple
                    '#1F51FF',  # neon blue
                    '#CCFF00',  # electric lime
                    '#FF073A',  # neon red
                    '#00FF9F',  # neon mint
                    '#FE4164',  # neon rose
                    '#04D9FF',  # electric blue
                    '#F4FF61',  # neon chartreuse
                    '#FF9933',  # neon tangerine
                    '#8AFF00',  # neon spring green
                    '#FA00FF',  # neon fuchsia
                    '#00FFEF',  # neon turquoise
                    '#FFB200',  # neon amber
                    '#D5FF00',  # neon citron
                ]

                def _tsne_scatter(ax, embedding, labels, title, cmap=None):
                    unique = sorted(set(labels))
                    n = max(len(unique), 1)
                    colors = [NEON_PALETTE[i % len(NEON_PALETTE)] for i in range(n)]
                    for color, label in zip(colors, unique):
                        mask = labels == label
                        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                                   c=[color], label=str(label),
                                   s=8, alpha=0.6, linewidths=0)
                    ax.set_title(title)
                    ax.set_xlabel('t-SNE 1')
                    ax.set_ylabel('t-SNE 2')
                    ax.legend(markerscale=2, fontsize=7,
                              loc='best', framealpha=0.7)

                # ── Plot 1: coloured by true cell type ────────────────────────
                fig, ax = plt.subplots(figsize=(7, 6))
                _tsne_scatter(ax, embedding, true_labels_plot,
                              't-SNE — True Cell Type', cmap='Set1')
                plt.tight_layout()
                p = os.path.join(OUTPUT_PATH, 'tsne_true_cell_type.png')
                plt.savefig(p, dpi=150)
                plt.close()
                log(f"  Saved: {p}")

                # ── Plot 2: coloured by predicted top cluster ─────────────────
                if top_arr is not None:
                    fig, ax = plt.subplots(figsize=(7, 6))
                    _tsne_scatter(ax, embedding, top_arr[:_n].astype(str),
                                  't-SNE — Predicted Top Cluster')
                    plt.tight_layout()
                    p = os.path.join(OUTPUT_PATH, 'tsne_top_cluster.png')
                    plt.savefig(p, dpi=150)
                    plt.close()
                    log(f"  Saved: {p}")

                # ── Plot 3: coloured by predicted middle cluster ───────────────
                if mid_arr is not None:
                    fig, ax = plt.subplots(figsize=(7, 6))
                    _tsne_scatter(ax, embedding, mid_arr[:_n].astype(str),
                                  't-SNE — Predicted Middle Cluster')
                    plt.tight_layout()
                    p = os.path.join(OUTPUT_PATH, 'tsne_middle_cluster.png')
                    plt.savefig(p, dpi=150)
                    plt.close()
                    log(f"  Saved: {p}")

                # ── Plot 4: side-by-side true vs top predicted ────────────────
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                _tsne_scatter(axes[0], embedding, true_labels_plot,
                              'True Cell Type', cmap='Set1')
                if top_arr is not None:
                    _tsne_scatter(axes[1], embedding, top_arr[:_n].astype(str),
                                  'Predicted Top Cluster')
                else:
                    axes[1].set_visible(False)
                plt.suptitle('t-SNE: True vs Predicted', fontsize=13)
                plt.tight_layout()
                p = os.path.join(OUTPUT_PATH, 'tsne_comparison.png')
                plt.savefig(p, dpi=150)
                plt.close()
                log(f"  Saved: {p}")

            except Exception as e:
                log(f"t-SNE generation failed: {e}")

    # ── Timing summary ─────────────────────────────────────────────────────────
    tracemalloc.stop()
    end_time  = time.perf_counter()
    elapsed   = end_time - start_time

    _step_times["  ↳ batch prep"]                  = (_batch_prep_s,     None)
    _step_times["  ↳ forward() total"]            = (_total_forward_s,  None)
    _step_times["    ↳ input norm"]               = (_input_norm_s,     None)
    _step_times["    ↳ edge index (dense→sparse)"]= (_edge_index_s,     None)
    _step_times["    ↳ GATE Encoder"]             = (_gate_encoder_s,   None)
    _step_times["    ↳ dot-product decode"]       = (_dot_product_s,    None)
    _step_times["    ↳ GATE Decoder"]             = (_gate_decoder_s,   None)
    _step_times["    ↳ clustering total"]         = (_clustering_s,     None)
    _step_times["      ↳ output layers (MLP)"]    = (_output_layers_s,  None)
    _step_times["      ↳ top community module"]   = (_top_comm_s,       None)
    _step_times["      ↳ select subsets"]         = (_select_subsets_s, None)
    _step_times["      ↳ middle community modules"]= (_middle_comm_s,   None)
    _step_times["  ↳ loss compute"]               = (_loss_compute_s,   None)
    _step_times["  ↳ backward pass"]              = (_backward_s,       None)
    _step_times["  ↳ grad clip"]                  = (_grad_clip_s,      None)
    _step_times["  ↳ optimizer step"]             = (_optimizer_step_s, None)
    _step_times["  ↳ gpu cleanup"]                = (_gpu_cleanup_s,    None)
    _step_times["  ↳ validation eval"]            = (_validation_s,     None)
    _step_times["  ↳ perf eval (periodic)"]       = (_perf_eval_s,      None)
    _step_times[f"  ↳ forward() calls"]           = (_forward_calls,    None)

    time_file = os.path.join(OUTPUT_PATH, "execution_time.txt")
    with open(time_file, "w") as _tf:
        _tf.write(f"Total execution time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
        _tf.write(f"Cell types: {TARGET_CELL_TYPES}\n")
        _tf.write(f"Final subset: {nodes} cells\n")
        _tf.write(f"World size: {WORLD_SIZE} processes\n\n")
        _tf.write(f"{'Step':<45} {'Time':>10}  {'Peak MB':>10}\n")
        _tf.write("-" * 70 + "\n")
        for _name, (_val, _mb) in _step_times.items():
            if _name == f"  ↳ forward() calls":
                _tf.write(f"{_name:<45} {int(_val):>10}\n")
            elif _name.startswith("  ↳") or _name.startswith("    ↳") or _name.startswith("      ↳"):
                _tf.write(f"{_name:<45} {_fmt(_val):>10}\n")
            else:
                _mb_str = f"{_mb:.1f}" if _mb is not None else "  n/a"
                _tf.write(f"{_name:<45} {_fmt(_val):>10}  {_mb_str:>10}\n")
        _tf.write("-" * 70 + "\n")
        _tf.write(f"{'TOTAL':<45} {_fmt(elapsed):>10}\n")
    log(f"Timing saved: {time_file}")

    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    log(f"Cell types trained on : {TARGET_CELL_TYPES}")
    log(f"Cells used            : {nodes:,} / {total_cells:,}")
    log(f"Community sizes       : {comm_sizes}")
    log(f"Results               : {OUTPUT_PATH}")
    log("=" * 80)

if WORLD_SIZE > 1:
    dist.destroy_process_group()

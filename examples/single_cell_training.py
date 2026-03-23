"""
Train DeepHCD on SUBSET of Converted Seurat RDS Data
Supports single-process and multi-node distributed training via torch.distributed.

Launch (single process):
    python -u single_cell_training.py

Launch (multi-node, torchrun — recommended):
    torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$GPUS_PER_NODE \\
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:29500 \\
        single_cell_training.py

Launch (multi-node, MPI):
    mpirun -n $TOTAL_PROCS python -u single_cell_training.py

SLURM job script tip: always set PYTHONUNBUFFERED=1 so output is not lost if
the job is killed before buffers flush.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import time
import tracemalloc

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

# ── Distributed setup ────────────────────────────────────────────────────────

def _init_distributed():
    """
    Initialize torch.distributed. Supports three launchers:
      - torchrun:  sets RANK + WORLD_SIZE automatically
      - srun/SLURM: sets SLURM_PROCID + SLURM_NTASKS
      - mpirun:    sets OMPI_COMM_WORLD_RANK
    Falls back to (rank=0, world_size=1) for plain `python` launches.
    """
    if not dist.is_available():
        return 0, 1

    # torchrun already sets RANK/WORLD_SIZE — use them directly
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend)

    # SLURM srun — use MPI backend which uses the cluster's interconnect fabric
    # (InfiniBand/OmniPath) rather than gloo's arbitrary TCP ports that HPC
    # firewalls typically block for inter-node all-reduce.
    elif 'SLURM_PROCID' in os.environ:
        dist.init_process_group(backend='mpi')

    # MPI launcher (mpirun / mpiexec)
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

def log(*args, **kwargs):
    """Print only from rank 0, always flushed (important for SLURM log files)."""
    if IS_MAIN:
        print(*args, **kwargs, flush=True)

# ── Timing / memory infrastructure ──────────────────────────────────────────

_step_times = {}

class StepTimer:
    """Context manager: times a step and, on rank 0, captures peak memory."""
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
    if seconds >= 60:
        return f"{seconds/60:.1f} min"
    return f"{seconds:.2f}s"

if IS_MAIN:
    tracemalloc.start()
start_time = time.perf_counter()

# ============================================================================
# SUBSET CONFIGURATION - SET THESE!
# ============================================================================

USE_SUBSET = True
SUBSET_SIZE = 500
SUBSET_METHOD = 'stratified'    # 'random' or 'stratified'
RANDOM_SEED = 42

# ============================================================================
# Configuration
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERTED_DATA_DIR = os.environ.get('DEEPHCD_DATA_DIR',
                                     os.path.join(_SCRIPT_DIR, 'converted_data'))
OUTPUT_PATH = os.environ.get('DEEPHCD_OUTPUT_DIR',
                              os.path.join(_SCRIPT_DIR, 'deephcd_subset_training'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_PCS = 50
K_NEIGHBORS = 30

if IS_MAIN:
    os.makedirs(OUTPUT_PATH, exist_ok=True)

log("=" * 80)
log("TRAINING DEEPHCD MODEL - MULTI-NODE DISTRIBUTED MODE")
log("=" * 80)
log(f"\nDevice: {DEVICE}")
log(f"World size: {WORLD_SIZE} processes")
log(f"Subset enabled: {USE_SUBSET}")
if USE_SUBSET:
    log(f"Target subset size: {SUBSET_SIZE} cells")
    log(f"Subset method: {SUBSET_METHOD}")

# ============================================================================
# STEPS 1 + 1.5 + 2a/2c: Load → Subset → PCA → Normalize
# Performed by rank 0 only. The resulting scaled feature matrix X_np is then
# broadcast to all other ranks. This avoids each rank loading gigabytes of
# raw expression data independently.
# ============================================================================

log("\n" + "=" * 80)
log("STEP 1: Loading Data  [rank 0 only]")
log("=" * 80)

def load_converted_data(data_dir):
    """Load data converted from RDS, keeping matrices sparse."""
    data = {}

    print("Loading expression matrix...", flush=True)
    expr_path = os.path.join(data_dir, "expression_matrix.mtx")
    expression = mmread(expr_path).tocsr()
    print(f"  Expression: {expression.shape}", flush=True)

    genes = pd.read_csv(os.path.join(data_dir, "genes.csv"))['gene'].tolist()
    cells = pd.read_csv(os.path.join(data_dir, "cells.csv"))['cell'].tolist()
    print(f"  Genes: {len(genes)}, Cells: {len(cells)}", flush=True)

    metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
    print(f"  Metadata: {metadata.shape}", flush=True)

    adj_path = os.path.join(data_dir, "adjacency_matrix.mtx")
    if os.path.exists(adj_path):
        print("Loading adjacency matrix (sparse)...", flush=True)
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

    data['expression'] = expression
    data['genes'] = genes
    data['cells'] = cells
    data['metadata'] = metadata
    data['adjacency'] = adjacency
    data['labels'] = labels
    return data

# Non-main ranks sit here until the broadcast below.
if IS_MAIN:
    with StepTimer("Step 1: Load Data"):
        data = load_converted_data(CONVERTED_DATA_DIR)

    # ── Step 1.5: Subset ──────────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("STEP 1.5: Applying Subset")
    log("=" * 80)

    with StepTimer("Step 1.5: Subset"):
        expression = data['expression']
        if expression.shape[0] < expression.shape[1]:
            log("Transposing expression matrix to (cells x genes)")
            expression = expression.T

        total_cells = expression.shape[0]
        log(f"Original data: {total_cells} cells × {len(data['genes'])} genes")

        if USE_SUBSET and total_cells > SUBSET_SIZE:
            log(f"\n SUBSETTING: {SUBSET_SIZE} / {total_cells} cells ({100*SUBSET_SIZE/total_cells:.1f}%)")
            np.random.seed(RANDOM_SEED)

            if SUBSET_METHOD == 'random':
                indices = np.sort(np.random.choice(total_cells, size=SUBSET_SIZE, replace=False))
            elif SUBSET_METHOD == 'stratified' and data['labels'] is not None:
                labels_arr = data['labels']
                unique_labels = np.unique(labels_arr)
                indices_list = []
                for label in unique_labels:
                    cluster_indices = np.where(labels_arr == label)[0]
                    n_from = max(1, int(SUBSET_SIZE * len(cluster_indices) / total_cells))
                    sampled = (np.random.choice(cluster_indices, size=n_from, replace=False)
                               if n_from < len(cluster_indices) else cluster_indices)
                    indices_list.append(sampled)
                    log(f"    Cluster {label}: {len(sampled)}/{len(cluster_indices)} cells")
                indices = np.sort(np.concatenate(indices_list))
            else:
                log("Falling back to random sampling...")
                indices = np.sort(np.random.choice(total_cells, size=SUBSET_SIZE, replace=False))

            expression = expression[indices, :]
            data['cells'] = [data['cells'][i] for i in indices]
            data['metadata'] = data['metadata'].iloc[indices].copy()
            if data['adjacency'] is not None:
                data['adjacency'] = data['adjacency'][indices][:, indices]
            if data['labels'] is not None:
                data['labels'] = data['labels'][indices]
            log(f"\n SUBSET APPLIED: {expression.shape[0]} cells × {expression.shape[1]} genes")
        else:
            log("Using full dataset")

        total_cells = expression.shape[0]

    # ── Step 2a: PCA (sparse-safe TruncatedSVD) ───────────────────────────────
    log("\n" + "=" * 80)
    log("STEP 2: PCA + Normalize  [rank 0 only — result broadcast to all ranks]")
    log("=" * 80)

    n_cells_local, n_genes = expression.shape
    log(f"Working with: {n_cells_local} cells × {n_genes} genes")

    if n_genes > N_PCS:
        with StepTimer("Step 2a: PCA"):
            log(f"\nApplying TruncatedSVD (sparse PCA): {n_genes} genes → {N_PCS} PCs")
            svd = TruncatedSVD(n_components=N_PCS, random_state=42)
            X_np = svd.fit_transform(expression)  # expression stays sparse
            log(f"  Variance explained: {svd.explained_variance_ratio_.sum():.1%}")
            log(f"  PCA embedding shape: {X_np.shape}")
    else:
        log("\nUsing raw expression (n_genes <= N_PCS)")
        X_np = expression.toarray() if issparse(expression) else expression

    with StepTimer("Step 2c: Normalize"):
        log("\nNormalizing features...")
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np).astype(np.float32)
        log("  Features normalized (mean=0, std=1)")

    # Build true_labels on rank 0
    true_labels = None
    if data['labels'] is not None:
        labels_arr = data['labels']
        unique_labels = np.unique(labels_arr)
        n_clusters = len(unique_labels)
        log(f"\nProcessing labels: {n_clusters} clusters")
        if n_clusters > 20:
            from sklearn.cluster import KMeans
            n_top = max(5, n_clusters // 4)
            log(f"  Creating 2-level hierarchy: {n_top} (top) / {n_clusters} (middle)")
            labels_top = KMeans(n_clusters=n_top, random_state=42, n_init=10).fit_predict(X_np)
            true_labels = [labels_top, labels_arr]
        else:
            log(f"  Using single level: {n_clusters} clusters")
            true_labels = [labels_arr]

    _broadcast_meta = [X_np.shape[0], X_np.shape[1], total_cells, true_labels]
else:
    X_np = None
    _broadcast_meta = [None, None, None, None]
    total_cells = 0

# ── Broadcast scaled PCA embedding from rank 0 to all ranks ──────────────────
# X_np is ~n_cells × 50 float32 (a few MB). Each rank then independently
# computes A from the same X_np — embarrassingly parallel graph construction.
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

# ── Step 2d: Build Graph  [all ranks in parallel] ─────────────────────────────
# All ranks received the same X_np, so get_input_graph is deterministic and
# produces the same A on every rank simultaneously — no serialisation needed.
log("\n" + "=" * 80)
log("STEP 2d: Build Graph  [all ranks in parallel]")
log("=" * 80)

with StepTimer(f"Step 2d: Build Graph [rank {RANK}]"):
    log("\nBuilding graph from PCA features...")
    A_graph, A_np = get_input_graph(X=X_np, method='Correlation', K=K_NEIGHBORS, metric='1-R^2')

log(f"  Graph: {A_graph.number_of_nodes()} nodes, {A_graph.number_of_edges()} edges")
log(f"  Adjacency density: {A_np.sum() / (X_np.shape[0]**2):.4f}")

A = torch.clamp(torch.FloatTensor(A_np) + torch.eye(X_np.shape[0]), 0, 1)
X = torch.FloatTensor(X_np)
nodes, features = X.shape
log(f"\nPrepared tensors [rank {RANK}]: X={X.shape}, A={A.shape}")

assert (not USE_SUBSET or total_cells <= SUBSET_SIZE or nodes >= SUBSET_SIZE * 0.9), \
    f"Subset failed! Expected ~{SUBSET_SIZE} nodes, got {nodes}"

# ============================================================================
# STEP 3: Estimate Community Sizes  [rank 0 computes, all ranks receive]
# ============================================================================

log("\n" + "=" * 80)
log("STEP 3: Estimating Community Sizes")
log("=" * 80)

_comm_list = [None]
if IS_MAIN:
    with StepTimer("Step 3: Community Size Estimation"):
        try:
            comm_middle, comm_top = compute_kappa(X, A, method='bethe_hessian', verbose=True)
            comm_sizes = [comm_middle, comm_top]
            log(f"Bethe-Hessian: Top={comm_top}, Middle={comm_middle}")
        except Exception as e:
            log(f"Bethe-Hessian failed: {e}")
            comm_top = max(10, nodes // 200)
            comm_middle = max(20, nodes // 50)
            comm_sizes = [comm_middle, comm_top]
            log(f"Using heuristic: Top={comm_top}, Middle={comm_middle}")
    _comm_list = [comm_sizes]

if WORLD_SIZE > 1:
    dist.broadcast_object_list(_comm_list, src=0)
comm_sizes = _comm_list[0]

# ============================================================================
# STEP 4: Create Model  [all ranks — identical architecture, wrapped in DDP]
# ============================================================================

log("\n" + "=" * 80)
log("STEP 4: Creating Model")
log("=" * 80)

if nodes < 1000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-3, 64, 200
    log(f"Small dataset (<1K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
elif nodes < 5000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-4, 128, 150
    log(f"Medium dataset (1-5K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
elif nodes < 10000:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-5, 256, 100
    log(f"Large dataset (5-10K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
else:
    LEARNING_RATE, BATCH_SIZE, EPOCHS = 1e-5, 512, 100
    log(f"Very large dataset (>10K): LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")

EARLY_STOPPING = True
PATIENCE = 10

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

    # DDP is intentionally disabled: the Trainer does its own random mini-batching
    # independently per rank, so ranks drift out of sync and deadlock on all-reduce.
    # Multi-node benefit comes from parallel data loading and graph construction above.
    if WORLD_SIZE > 1:
        log("  Note: DDP skipped — Trainer not compatible with gradient synchronisation")

n_params = sum(p.numel() for p in model.parameters())
log(f"Model: {n_params:,} parameters")

# ============================================================================
# STEP 5: Train  [all ranks — DDP synchronises gradients automatically]
# ============================================================================

log("\n" + "=" * 80)
log("STEP 5: Training Model")
log("=" * 80)

# Only rank 0 trains — other ranks finished their job during parallel graph building
if IS_MAIN:
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

    log("Starting training...\n")
    reset_forward_timing()
    with StepTimer("Step 5: Training"):
        output = trainer.fit(DEVICE)

    _gate_encoder_s = forward_timing['gate_encoder']
    _gate_decoder_s = forward_timing['gate_decoder']
    _clustering_s   = forward_timing['clustering']
    _forward_calls  = forward_timing['calls']

# ============================================================================
# STEPS 6 & 7: Save Results + Heatmaps  [rank 0 only]
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
            log("\nPerformance Metrics:")
            for i, perf in enumerate(final_perf):
                if perf is not None:
                    layer = 'Top' if i == 0 else 'Middle'
                    log(f"  {layer}: H={perf[0]:.3f}, C={perf[1]:.3f}, NMI={perf[2]:.3f}, ARI={perf[3]:.3f}")

    with StepTimer("Step 6: Save Results"):
        # Unwrap DDP to get the underlying module for state_dict
        save_model = model.module if isinstance(model, DDP) else model
        MODEL_PATH = os.path.join(OUTPUT_PATH, 'trained_model.pth')
        torch.save({
            'model_state_dict': save_model.state_dict(),
            'comm_sizes': comm_sizes,
            'subset_size': nodes,
            'config': {
                'method': 'top_down',
                'ae_hidden_dims': [128, 64, 32],
                'comm_operator': 'Linear',
                'dropout': 0.3
            }
        }, MODEL_PATH)
        log(f"\n✓ Model saved: {MODEL_PATH}")

        if hasattr(output, 'predicted_train'):
            pred_path = os.path.join(OUTPUT_PATH, 'predictions.csv')
            top_preds = output.predicted_train['top'].cpu().numpy() if 'top' in output.predicted_train else None
            mid_preds = output.predicted_train['middle'].cpu().numpy() if 'middle' in output.predicted_train else None
            n_preds = (len(top_preds) if top_preds is not None else
                       len(mid_preds) if mid_preds is not None else len(data['cells']))
            pred_df = pd.DataFrame({
                'cell': data['cells'][:n_preds],
                'top_cluster': top_preds if top_preds is not None else [None] * n_preds,
                'middle_cluster': mid_preds if mid_preds is not None else [None] * n_preds,
            })
            pred_df.to_csv(pred_path, index=False)
            log(f"✓ Predictions saved: {pred_path}")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    with StepTimer("Step 7: Heatmaps"):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns

            if not hasattr(output, 'predicted_train'):
                pred_df = pd.read_csv(pred_path)
                _n = len(pred_df)
            else:
                _top = output.predicted_train['top'].cpu().numpy() if 'top' in output.predicted_train else None
                _mid = output.predicted_train['middle'].cpu().numpy() if 'middle' in output.predicted_train else None
                _n = (len(_top) if _top is not None else
                      len(_mid) if _mid is not None else len(data['cells']))
                pred_df = pd.DataFrame({
                    'cell': data['cells'][:_n],
                    'top_cluster': _top if _top is not None else [None] * _n,
                    'middle_cluster': _mid if _mid is not None else [None] * _n,
                })

            X_np_plot = X.numpy()[:_n]

            sorted_df = pred_df.dropna(subset=['top_cluster']).sort_values(['top_cluster', 'middle_cluster'])
            sorted_idx = sorted_df.index.to_numpy()
            corr_matrix = np.corrcoef(X_np_plot[sorted_idx])

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
            log(f"✓ Cell-cell correlation heatmap saved: {heatmap1_path}")

            for cluster_level, cluster_col in [('top', 'top_cluster'), ('middle', 'middle_cluster')]:
                co = pred_df.dropna(subset=[cluster_col])
                if len(co) == 0:
                    continue
                cluster_ids = sorted(co[cluster_col].astype(int).unique())
                cluster_means = np.stack([
                    X_np_plot[co[co[cluster_col].astype(int) == c].index].mean(axis=0)
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
                log(f"✓ Cluster correlation heatmap saved: {heatmap2_path}")

        except Exception as e:
            log(f"⚠ Heatmap generation failed: {e}")

    # ── Timing & memory summary ────────────────────────────────────────────────
    tracemalloc.stop()
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    _step_times["  ↳ GATE Encoder (total)"] = (_gate_encoder_s, None)
    _step_times["  ↳ GATE Decoder (total)"] = (_gate_decoder_s, None)
    _step_times["  ↳ Clustering (total)"]   = (_clustering_s,   None)
    _step_times[f"  ↳ forward() calls"]     = (_forward_calls,  None)

    time_file = os.path.join(OUTPUT_PATH, "execution_time.txt")
    with open(time_file, "w") as _tf:
        _tf.write(f"Total execution time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
        _tf.write(f"Subset size: {nodes} cells ({SUBSET_METHOD})\n")
        _tf.write(f"World size: {WORLD_SIZE} processes\n\n")
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
    log(f"✓ Timing saved: {time_file}")

    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    log(f"✓ World size: {WORLD_SIZE} processes")
    log(f"✓ Trained on {nodes} / {total_cells} cells ({100*nodes/total_cells:.1f}%)")
    log(f"\n{'Step':<45} {'Time':>10}  {'Peak MB':>10}")
    log("-" * 70)
    for _name, (_val, _mb) in _step_times.items():
        if _name.startswith("  ↳ forward"):
            log(f"  {'↳ forward() calls':<43} {int(_val):>10}")
        elif _name.startswith("  ↳"):
            log(f"{_name:<45} {_fmt(_val):>10}")
        else:
            _mb_str = f"{_mb:.1f}" if _mb is not None else "  n/a"
            log(f"{_name:<45} {_fmt(_val):>10}  {_mb_str:>10}")
    log("-" * 70)
    log(f"{'TOTAL':<45} {_fmt(elapsed):>10}")
    log(f"\n✓ Results: {OUTPUT_PATH}")
    log("=" * 80)

if WORLD_SIZE > 1:
    dist.destroy_process_group()

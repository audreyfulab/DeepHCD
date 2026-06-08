#!/usr/bin/env python3
"""
Reproduce Tables 4.2 / 4.3 of Kvamme thesis: compare DeepHCD against
Ward-linkage hierarchical clustering (HC) and the Louvain method on
simulated hierarchical networks.

Modeled after `run_simulations_utils.py` in the older HGRN_repo, but
ported to the DeepHCD_copy package layout. Settings come from thesis
sections 4.3 / 4.3.4 / 4.3.5 and the captions of Tables 4.2 and 4.3.

Default scenario: Table 4.2 balanced, fully-connected, small-world,
sigma=0.1, feature matrix as input.

Usage:
    python run_simulations.py \
        --topology small_world --input-type feature \
        --n-replicates 3 --output-dir ./table42_run/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import networkx as nx

from sklearn.cluster import AgglomerativeClustering
from networkx.algorithms.community import louvain_communities

from deephcd.simulate.simulate import simulate_graph
from deephcd.model.model import HCD
from deephcd.model.train import Trainer
from deephcd.utils.utilities import LoadData, compute_kappa, node_clust_eval
from deephcd.utils.train_utils import split_dataset


# ---------------------------------------------------------------------------
# Simulation settings  (thesis Tables 4.2 / 4.3, section 4.3)
# ---------------------------------------------------------------------------

TOPOLOGY_MAP = {
    "small_world": "small world",
    "scale_free":  "scale free",
    "random":      "random graph",
}


def make_sim_args(
    *,
    savepath: str,
    topology: str = "small world",
    connect: str = "full",
    balanced: bool = True,
    sd: float = 0.1,
    seed: int = 42,
) -> argparse.Namespace:
    """Thesis-matched defaults: 5 -> 15 -> 300 hierarchy, 500 samples."""
    a = argparse.Namespace()

    a.connect           = connect
    a.top_layer_nodes   = 5
    a.nodes_per_super2  = (3, 3)
    a.nodes_per_super3  = (20, 20) if balanced else (10, 30)
    a.layers            = 3

    a.subgraph_type     = topology
    a.subgraph_prob     = [0.5, 0.3]
    a.node_degree_middle = 2
    a.node_degree_bottom = 6

    a.connect_prob_middle = [0.1, 0.05]
    a.connect_prob_bottom = [0.15, 0.02]

    a.sample_size = 150
    a.SD          = sd
    a.common_dist = False

    a.use_weighted_graph  = False
    a.within_edgeweights  = (0.5, 0.8)
    a.between_edgeweights = (0.0, 0.2)
    a.force_connect       = True
    a.mixed_graph         = False

    a.set_seed    = True
    a.seed_number = seed

    a.save_pdf = False
    a.save_png = False

    a.savepath = savepath
    return a


# ---------------------------------------------------------------------------
# Adjacency estimation  (thesis section 4.3.4, eq. 4.25:  r > 0.2)
# ---------------------------------------------------------------------------

def estimated_adjacency(features: np.ndarray, r_cutoff: float = 0.2) -> np.ndarray:
    corr = np.corrcoef(features)
    A = (np.abs(corr) > r_cutoff).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return A


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_ward(features: np.ndarray, k_top: int, k_mid: int) -> Tuple[np.ndarray, np.ndarray]:
    top = AgglomerativeClustering(n_clusters=k_top, metric="euclidean",
                                  linkage="ward").fit(features).labels_
    mid = AgglomerativeClustering(n_clusters=k_mid, metric="euclidean",
                                  linkage="ward").fit(features).labels_
    return top, mid


def run_louvain(A: np.ndarray, seed: int = 0) -> np.ndarray:
    G = nx.from_numpy_array(A)
    communities = louvain_communities(G, seed=seed)
    labels = np.zeros(A.shape[0], dtype=int)
    for cid, members in enumerate(communities):
        for n in members:
            labels[n] = cid
    return labels


def run_deephcd(
    X: torch.Tensor,
    A: torch.Tensor,
    labels_list: List[torch.Tensor],
    comm_sizes: List[int],
    output_path: str,
    device: str,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """DeepHCD-NOL configuration matching section 4.3.5.

    Trains on an 80/20 split, then re-runs the trained model on the *full*
    feature/adjacency tensors so the returned predictions are aligned with
    the full sorted-truth label arrays.
    """
    train_set, val_set = split_dataset(X, A, labels_list)  # 80/20 split

    nodes, features = X.shape
    model = HCD(
        nodes=nodes,
        attrib=features,
        method="top_down",
        ae_hidden_dims=[256, 128],          # thesis: encoder/decoder 256, bottleneck 128
        comm_sizes=comm_sizes,
        ae_operator="GATv2Conv",
        dropout=0.2,
        normalize_input=True,
    ).to(device)

    trainer = Trainer(
        model=model,
        X=train_set[0],
        A=train_set[1],
        validation_data=val_set,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        early_stopping=True,
        patience=15,
        true_labels=train_set[2],
        output_path=output_path,
        verbose=False,
        # gamma, delta, _lambda all default to 1 in Trainer (eq. 4.17 with unit weights)
    )
    trainer.fit(device)

    # Re-run the trained model on the full graph so predictions are
    # comparable to the full sorted-truth arrays (HC and Louvain see the
    # full graph too).
    model.eval()
    with torch.no_grad():
        forward = model(X.to(device), A.to(device))
    S_all = forward[6]  # (top_assignments, middle_assignments)
    pred_top = S_all[0].detach().cpu().numpy().astype(int)
    pred_mid = S_all[1].detach().cpu().numpy().astype(int)
    return pred_top, pred_mid


# ---------------------------------------------------------------------------
# Metrics aggregation  (matches Tables 4.2 / 4.3 layout)
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    method: str
    top_metrics: np.ndarray    # [H, C, NMI, ARI]
    mid_metrics: np.ndarray


def evaluate_method(name: str,
                    top_pred: np.ndarray, mid_pred: np.ndarray,
                    top_true: np.ndarray, mid_true: np.ndarray) -> MethodResult:
    top = node_clust_eval(top_true, top_pred, verbose=False)
    mid = node_clust_eval(mid_true, mid_pred, verbose=False)
    return MethodResult(method=name, top_metrics=top, mid_metrics=mid)


def build_table(results_per_topology: dict[str, List[MethodResult]],
                layer: str = "middle") -> pd.DataFrame:
    """Build a Table-4.2-style DataFrame from accumulated per-replicate results.

    `results_per_topology[topology]` is a list of MethodResult averaged across
    replicates (one entry per method, in order).
    """
    rows = []
    for topology, results in results_per_topology.items():
        for r in results:
            m = r.mid_metrics if layer == "middle" else r.top_metrics
            H, C, NMI, ARI = m[0], m[1], m[2], m[3]
            rows.append({
                "Topology":    topology,
                "Method":      r.method,
                "Completeness (C)": round(float(C), 3),
                "Homogeneity (H)":  round(float(H), 3),
                "NMI":              round(float(NMI), 3),
                "ARI":              round(float(ARI), 3),
                "|C-H|":            round(abs(float(C) - float(H)), 3),
            })
    return pd.DataFrame(rows)


def aggregate_replicates(per_rep: List[MethodResult]) -> MethodResult:
    top = np.mean(np.stack([r.top_metrics for r in per_rep], axis=0), axis=0)
    mid = np.mean(np.stack([r.mid_metrics for r in per_rep], axis=0), axis=0)
    return MethodResult(method=per_rep[0].method, top_metrics=top, mid_metrics=mid)


# ---------------------------------------------------------------------------
# Main pipeline for one (topology, replicate) cell
# ---------------------------------------------------------------------------

def run_one_replicate(
    *,
    topology_key: str,
    replicate_idx: int,
    base_output_dir: str,
    args,
) -> List[MethodResult]:
    """Simulate one network and score Louvain / HC-Ward / DeepHCD on it."""
    topology = TOPOLOGY_MAP[topology_key]
    seed = args.seed_base + replicate_idx
    rep_dir = os.path.join(base_output_dir, topology_key, f"rep_{replicate_idx:02d}")
    os.makedirs(rep_dir, exist_ok=True)

    # 1. Simulate
    sim_args = make_sim_args(
        savepath = rep_dir + "/",
        topology = topology,
        connect  = args.connect,
        balanced = args.balanced,
        sd       = args.sd,
        seed     = seed,
    )
    print(f"[{topology_key} rep {replicate_idx}] simulating...")
    simulate_graph(sim_args)

    # 2. Load + sort
    pe, true_adj, idx_top, idx_mid, _, sorted_top, sorted_mid = LoadData(rep_dir)

    # Input feature matrix (genes x samples) sorted by middle-layer labels.
    features_sorted = pe[idx_mid, :]
    if args.input_type == "correlation":
        X_np = np.corrcoef(features_sorted)
    else:
        X_np = features_sorted
    X = torch.FloatTensor(X_np)

    # Adjacency: estimated from correlations as in section 4.3.4.
    A_np = estimated_adjacency(features_sorted, r_cutoff=args.r_cutoff)
    A = torch.FloatTensor(A_np) + torch.eye(A_np.shape[0])

    top_true = np.asarray(sorted_top)
    mid_true = np.asarray(sorted_mid)

    # 3. Community-count estimates (Bethe-Hessian; thesis section 4.4.3).
    # compute_kappa returns (kappa_middle, kappa_top).
    if args.use_true_k:
        k_mid_est, k_top_est = 15, 5
        print("   using true k: k_mid=15, k_top=5")
    else:
        k_mid_est, k_top_est = compute_kappa(X, A, method="bethe_hessian", verbose=False)
        print(f"   Bethe-Hessian k estimate: k_mid={k_mid_est}, k_top={k_top_est}")

    # Clamp to >= 2 so AgglomerativeClustering and HCD don't degenerate.
    k_mid = max(int(k_mid_est), 2)
    k_top = max(int(k_top_est), 2)
    comm_sizes = (k_mid, k_top)  # HCD(method='top_down') reverses internally

    # 4. Louvain  (uses estimated adjacency only, no k)
    louv_labels = run_louvain(A_np, seed=seed)

    # 5. Ward HC  (uses features only, k from Bethe)
    ward_top, ward_mid = run_ward(X_np, k_top=k_top, k_mid=k_mid)

    # 6. DeepHCD
    labels_list = [torch.LongTensor(sorted_top), torch.LongTensor(sorted_mid)]
    deep_top, deep_mid = run_deephcd(
        X, A, labels_list, comm_sizes,
        output_path=rep_dir + "/deephcd/",
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    # 7. Score every method at top + middle layers.
    return [
        evaluate_method("HC",       ward_top,    ward_mid,    top_true, mid_true),
        evaluate_method("DeepHCD",  deep_top,    deep_mid,    top_true, mid_true),
        evaluate_method("Louvain",  louv_labels, louv_labels, top_true, mid_true),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./table42_run_avg10/")
    parser.add_argument("--topology", nargs="+",
                        choices=list(TOPOLOGY_MAP),
                        default=list(TOPOLOGY_MAP))
    parser.add_argument("--input-type", choices=["feature", "correlation"],
                        default="feature",
                        help="DeepHCD input X: raw features vs gene-gene correlation matrix")
    parser.add_argument("--connect", choices=["full", "disc"], default="full",
                        help="Top-layer connectivity (Tables 4.2/4.3 use 'full')")
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Table 4.2 = balanced; Table 4.3 = --no-balanced")
    parser.add_argument("--sd", type=float, default=0.1,
                        help="Gene-expression noise sigma (thesis uses 0.1 and 0.5)")
    parser.add_argument("--r-cutoff", type=float, default=0.2)
    parser.add_argument("--n-replicates", type=int, default=10,
                        help="25 in the thesis; default 1 for a quick run")
    parser.add_argument("--use-true-k", action="store_true",
                        help="Use true k_top=5, k_mid=15 instead of Bethe-Hessian estimate")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("=" * 72)
    print("DeepHCD vs HC (Ward) vs Louvain  --  Tables 4.2 / 4.3 reproduction")
    print("=" * 72)
    print(f"  topologies:   {args.topology}")
    print(f"  input type:   {args.input_type}")
    print(f"  connect:      {args.connect}    balanced={args.balanced}    sd={args.sd}")
    print(f"  replicates:   {args.n_replicates}    device={args.device}")
    print()

    t0 = time.perf_counter()
    averaged: dict[str, List[MethodResult]] = {}

    for topo in args.topology:
        per_rep_by_method: dict[str, List[MethodResult]] = {"HC": [], "DeepHCD": [], "Louvain": []}
        for rep in range(args.n_replicates):
            try:
                results = run_one_replicate(
                    topology_key=topo,
                    replicate_idx=rep,
                    base_output_dir=args.output_dir,
                    args=args,
                )
            except Exception as e:
                import traceback
                print(f"   ! replicate {rep} failed: {e}", file=sys.stderr)
                traceback.print_exc()
                continue
            for r in results:
                per_rep_by_method[r.method].append(r)

        averaged[topo] = [aggregate_replicates(per_rep_by_method[m])
                          for m in ("HC", "DeepHCD", "Louvain")
                          if per_rep_by_method[m]]

    middle_table = build_table(averaged, layer="middle")
    top_table    = build_table(averaged, layer="top")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 72)
    print(f"DONE in {elapsed:.1f}s")
    print("=" * 72)
    print("\n--- Middle layer (matches Tables 4.2 / 4.3) ---")
    print(middle_table.to_string(index=False))
    print("\n--- Top layer ---")
    print(top_table.to_string(index=False))

    middle_csv = os.path.join(args.output_dir, "results_middle.csv")
    top_csv    = os.path.join(args.output_dir, "results_top.csv")
    middle_table.to_csv(middle_csv, index=False)
    top_table.to_csv(top_csv, index=False)
    print(f"\nSaved: {middle_csv}")
    print(f"Saved: {top_csv}")


if __name__ == "__main__":
    main()

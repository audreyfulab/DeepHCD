#!/usr/bin/env python3
"""
Minimal Example: Quick Hierarchical Network Simulation

This is a minimal working example that generates a small 3-layer hierarchy
and saves the results. Perfect for getting started quickly.
"""

import argparse
import os
import numpy as np
import pandas as pd
from deephcd.simulate.simulate import simulate_graph

ZERO_FRACTION = 0.8


# Create minimal configuration
args = argparse.Namespace()

# Network structure (creates ~5 → 15 → 150 nodes across 3 layers)
args.connect = 'disc'                    # Disconnected top layer
args.top_layer_nodes = 5                 # 5 top communities
args.nodes_per_super2 = (3, 3)          # 3 nodes per top community → 15 middle nodes
args.nodes_per_super3 = (10, 10)        # 10-12 nodes per middle node → ~150 bottom nodes
args.layers = 3                          # 3-layer hierarchy

# Subgraph topology
args.subgraph_type = 'small world'
args.subgraph_prob = [0.5, 0.3]         # Edge probability [middle, bottom]
args.node_degree_middle = 2             # Lower degree for middle layer
args.node_degree_bottom = 6             # Degree for bottom layer (must be < min nodes_per_super3)

# Connection probabilities
args.connect_prob_middle = [0.1, 0.05]  # [within, between] communities
args.connect_prob_bottom = [0.15, 0.02]

# Data generation
args.sample_size = 150                   # 200 samples/observations
args.SD = 0.1                           # Standard deviation
args.common_dist = False                # Different distributions per parent

# Graph properties
args.use_weighted_graph = False
args.within_edgeweights = (0.5, 0.8)
args.between_edgeweights = (0.0, 0.2)
args.force_connect = True
args.mixed_graph = False

# Reproducibility
args.set_seed = True
args.seed_number = 42

# Output
args.savepath = './1k_graph/'

# Create output directory
os.makedirs(args.savepath, exist_ok=True)

# Run simulation
print("Simulating hierarchical network...")
pe, gexp, nodes, edges, nx_all, adj_all, path, labels, ori = simulate_graph(args)

print("\nSimulation complete!")
print(f"   Network: {nodes[0]} → {nodes[1]} → {nodes[2]} nodes")
print(f"   Data: {pe.shape[0]} nodes × {pe.shape[1]} samples")

# Simulate genetic-data sparsity by zeroing a random fraction of entries
rng = np.random.default_rng(args.seed_number if args.set_seed else None)
dropout_mask = rng.random(pe.shape) < ZERO_FRACTION
pe[dropout_mask] = 0.0
sparsity = float((pe == 0).mean())
print(f"   Applied dropout: zeroed {ZERO_FRACTION:.0%} of entries (observed sparsity: {sparsity:.2%})")

# Overwrite the files written by simulate_graph so downstream loaders see the sparsified data
gexp_sparse = pd.DataFrame(data=np.transpose(pe), index=range(args.sample_size))
gexp_sparse.to_csv(args.savepath + '_gexp.csv')
np.save(args.savepath + '_gexp.npy', gexp_sparse.to_numpy())

print(f"   Saved to: {args.savepath}")

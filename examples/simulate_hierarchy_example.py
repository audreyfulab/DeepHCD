#!/usr/bin/env python3
"""
Example: Simulating Hierarchical Networks with Node-Level Data

This script demonstrates how to generate a 3-layer hierarchical network and
simulate pseudo-expression data at the individual nodes in the bottom layer.

The hierarchy structure:
- Top layer: Community-level (e.g., tissue types)
- Middle layer: Sub-community level (e.g., cell types)
- Bottom layer: Individual nodes (e.g., genes) with simulated expression data

Output files:
- Network topology at each layer (top, middle, bottom)
- Adjacency matrices
- Pseudo-expression data matrix (nodes x samples)
- Node labels showing hierarchical community assignments
- Visualization plots (heatmaps, network graphs)
"""

import os
import argparse
import numpy as np
import pandas as pd
from deephcd.simulate.simulate import simulate_graph


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Simulate Hierarchical Network')

    # Network topology settings
    parser.add_argument('--connect', default='disc', choices=['disc', 'full'],
                       help='Top layer connectivity: disc=disconnected, full=fully connected')
    parser.add_argument('--top_layer_nodes', type=int, default=5,
                       help='Number of top-level communities')
    parser.add_argument('--nodes_per_super2', nargs='+', type=int, default=[3, 3],
                       help='Min/max offspring per top-layer node (middle layer)')
    parser.add_argument('--nodes_per_super3', nargs='+', type=int, default=[10, 15],
                       help='Min/max offspring per middle-layer node (bottom layer)')

    # Subgraph structure
    parser.add_argument('--subgraph_type', default='small world',
                       help='Type of subgraph topology (small world, scale free, random)')
    parser.add_argument('--subgraph_prob', nargs='+', type=float, default=[0.5, 0.3],
                       help='Edge probability within subgraphs [middle, bottom]')
    parser.add_argument('--node_degree_middle', type=int, default=3,
                       help='Node degree for middle layer')
    parser.add_argument('--node_degree_bottom', type=int, default=8,
                       help='Node degree for bottom layer')

    # Connection probabilities between communities
    parser.add_argument('--connect_prob_middle', nargs='+', type=float, default=[0.1, 0.05],
                       help='Connection probability [within, between] communities in middle layer')
    parser.add_argument('--connect_prob_bottom', nargs='+', type=float, default=[0.15, 0.02],
                       help='Connection probability [within, between] communities in bottom layer')

    # Pseudo-expression data settings
    parser.add_argument('--sample_size', type=int, default=500,
                       help='Number of samples (e.g., cells, individuals)')
    parser.add_argument('--SD', type=float, default=0.1,
                       help='Standard deviation for simulated node values')
    parser.add_argument('--common_dist', action='store_true',
                       help='Use common N(0,σ) distribution for all parent nodes')

    # Network properties
    parser.add_argument('--layers', type=int, default=3, choices=[2, 3],
                       help='Number of hierarchical layers (2 or 3)')
    parser.add_argument('--use_weighted_graph', action='store_true',
                       help='Generate weighted edges')
    parser.add_argument('--within_edgeweights', nargs=2, type=float, default=[0.5, 0.8],
                       help='Min/max edge weights within communities')
    parser.add_argument('--between_edgeweights', nargs=2, type=float, default=[0.0, 0.2],
                       help='Min/max edge weights between communities')
    parser.add_argument('--force_connect', action='store_true', default=True,
                       help='Ensure at least one edge between connected communities')
    parser.add_argument('--mixed_graph', action='store_true',
                       help='Mix different subgraph topologies')

    # Reproducibility
    parser.add_argument('--set_seed', action='store_true', default=True,
                       help='Set random seed for reproducibility')
    parser.add_argument('--seed_number', type=int, default=42,
                       help='Random seed value')

    # Output settings
    parser.add_argument('--savepath', type=str, default='./simulated_hierarchy/',
                       help='Directory to save simulation outputs')

    args = parser.parse_args()

    # Convert list arguments to tuples for compatibility
    args.nodes_per_super2 = tuple(args.nodes_per_super2)
    args.nodes_per_super3 = tuple(args.nodes_per_super3)
    args.subgraph_prob = tuple(args.subgraph_prob)
    args.within_edgeweights = tuple(args.within_edgeweights)
    args.between_edgeweights = tuple(args.between_edgeweights)

    # Create output directory
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
        print(f"Created output directory: {args.savepath}")

    # Display simulation parameters
    print("="*70)
    print("HIERARCHICAL NETWORK SIMULATION")
    print("="*70)
    print(f"\nNetwork Structure:")
    print(f"  Layers: {args.layers}")
    print(f"  Top layer: {args.top_layer_nodes} communities ({args.connect})")
    print(f"  Middle layer: ~{args.top_layer_nodes * args.nodes_per_super2[0]} nodes")
    if args.layers == 3:
        expected_bottom = args.top_layer_nodes * args.nodes_per_super2[0] * args.nodes_per_super3[0]
        print(f"  Bottom layer: ~{expected_bottom} nodes")

    print(f"\nData Generation:")
    print(f"  Samples: {args.sample_size}")
    print(f"  Standard deviation: {args.SD}")
    print(f"  Common distribution: {args.common_dist}")

    print(f"\nOutput location: {args.savepath}")
    print("="*70)

    # Run simulation
    print("\nGenerating hierarchical network...")
    results = simulate_graph(args)

    # Unpack results
    pe, gexp, nodes_by_layer, edges_by_layer, nx_all, adj_all, path, ts_full, ori_nodes = results

    # Display results
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"\nNetwork Statistics:")
    for i, (n, e) in enumerate(zip(nodes_by_layer, edges_by_layer)):
        layer_name = ["Top", "Middle", "Bottom"][i] if args.layers == 3 else ["Top", "Bottom"][i]
        print(f"  {layer_name} layer: {n} nodes, {e} edges")

    print(f"\nData Matrix:")
    print(f"  Shape: {pe.shape} (nodes × samples)")
    print(f"  Total data points: {pe.size:,}")

    print(f"\nOutput Files Generated:")
    print(f"  Directory: {args.savepath}")
    print(f"    ├── top_layer_graph.pdf/png - Top layer network visualization")
    if args.layers == 3:
        print(f"    ├── middle_layer_graph.pdf/png - Middle layer network visualization")
    print(f"    ├── bottom_layer_graph.pdf/png - Bottom layer network visualization")
    print(f"    ├── heatmaps.pdf/png - Data correlation and adjacency heatmaps")
    print(f"    ├── _gexp.csv - Pseudo-expression data (CSV format)")
    print(f"    ├── _gexp.npy - Pseudo-expression data (NumPy format)")
    print(f"    ├── .npz - Network adjacency matrices and metadata")
    print(f"    └── directed_graphs.pkl - Network topology (pickle format)")

    print("\n" + "="*70)

    # Display sample of generated data
    print("\nSample of generated pseudo-expression data:")
    print(gexp.head())

    print(f"\nTo load this data later, use:")
    print(f"  from deephcd.utils.utilities import LoadData")
    print(f"  pe, adj, idx_top, idx_mid, labels, ... = LoadData('{args.savepath}')")

    return results


if __name__ == "__main__":
    main()

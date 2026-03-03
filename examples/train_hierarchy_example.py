#!/usr/bin/env python3
"""
Comprehensive Example: Train DeepHCD Model with Full Options

This example demonstrates the complete training workflow with:
- Data loading and preprocessing
- Train/validation splitting
- Model configuration
- Training with validation
- Results visualization and evaluation
"""

import argparse
import os
import torch
import numpy as np
from deephcd.model.model import HCD
from deephcd.model.train import Trainer
from deephcd.utils.train_utils import split_dataset
from deephcd.utils.utilities import LoadData, compute_kappa, get_input_graph


def main():
    parser = argparse.ArgumentParser(description='Train DeepHCD Model')

    # Data settings
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to simulated network data directory')
    parser.add_argument('--use_true_graph', action='store_true', default=True,
                       help='Use true adjacency matrix from simulation')
    parser.add_argument('--correlation_cutoff', type=float, default=0.2,
                       help='Correlation threshold for graph construction (if not using true graph)')

    # Model architecture
    parser.add_argument('--method', type=str, default='top_down', choices=['top_down', 'bottom_up'],
                       help='Hierarchical detection method')
    parser.add_argument('--ae_hidden_dims', nargs='+', type=int, default=[256, 128],
                       help='Hidden dimensions for graph autoencoder')
    parser.add_argument('--ae_operator', type=str, default='GATv2Conv',
                       choices=['GATConv', 'GATv2Conv', 'SAGEConv'],
                       help='Graph neural network operator')
    parser.add_argument('--attn_heads', type=int, default=5,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--normalize_input', action='store_true', default=True,
                       help='Normalize input features')

    # Community detection
    parser.add_argument('--compute_optimal_clusters', action='store_true', default=True,
                       help='Automatically estimate optimal number of communities')
    parser.add_argument('--kappa_method', type=str, default='bethe_hessian',
                       choices=['bethe_hessian', 'elbow', 'silouette'],
                       help='Method for estimating optimal communities')
    parser.add_argument('--comm_sizes', nargs=2, type=int, default=None,
                       help='Manual community sizes [middle, top] (overrides auto-detection)')

    # Training settings
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--use_batch_learning', action='store_true', default=True,
                       help='Use batch learning (vs full batch)')

    # Loss hyperparameters
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Feature reconstruction loss weight')
    parser.add_argument('--delta', type=float, default=10.0,
                       help='Modularity loss weight')
    parser.add_argument('--lambda_', nargs=2, type=float, default=[1/60, 1/20],
                       help='Clustering loss weights [middle, top]')

    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience for early stopping')

    # Data splitting
    parser.add_argument('--split_data', action='store_true', default=False,
                       help='Split data into train/validation sets')
    parser.add_argument('--train_val_split', nargs=2, type=float, default=[0.8, 0.2],
                       help='Train/validation split ratio')

    # Output settings
    parser.add_argument('--output_path', type=str, default='./training_output/',
                       help='Directory to save training outputs')
    parser.add_argument('--save_model', action='store_true', default=False,
                       help='Save trained model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed training information')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    print("="*80)
    print("DEEPHCD MODEL TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data_path}")
    print(f"  Device: {device}")
    print(f"  Method: {args.method}")
    print(f"  Architecture: {args.ae_operator} with hidden dims {args.ae_hidden_dims}")
    print(f"  Training: {args.epochs} epochs, batch size {args.batch_size}, lr {args.learning_rate}")
    print("="*80)

    # Load data
    print("\nLoading data...")
    pe, adj, idx_top, idx_mid, labels, sorted_top, sorted_mid = LoadData(args.data_path)

    # Prepare features
    X = torch.FloatTensor(pe[idx_mid, :])

    # Prepare adjacency matrix
    if args.use_true_graph:
        print("   Using true adjacency matrix from simulation")
        A = torch.FloatTensor(adj[:X.shape[0], :X.shape[0]]) + torch.eye(X.shape[0])
    else:
        print(f"   Constructing graph from correlations (cutoff={args.correlation_cutoff})")
        _, in_adj = get_input_graph(pe[idx_mid, :], method='Correlation', r_cutoff=args.correlation_cutoff)
        A = torch.FloatTensor(in_adj) + torch.eye(X.shape[0])

    nodes, features = X.shape
    true_labels = [sorted_top, sorted_mid]

    print(f"   Loaded: {nodes} nodes, {features} features")
    print(f"   True communities: Top={len(set(sorted_top))}, Middle={len(set(sorted_mid))}")

    # Split data if requested
    validation_data = None
    if args.split_data:
        print(f"\nSplitting data ({args.train_val_split[0]:.0%} train, {args.train_val_split[1]:.0%} validation)...")
        train, val_set = split_dataset(X, A, true_labels, args.train_val_split)
        X, A, true_labels = train
        validation_data = val_set
        print(f"   Train: {X.shape[0]} nodes, Validation: {val_set[0].shape[0]} nodes")

    # Compute optimal communities
    if args.compute_optimal_clusters:
        print(f"\nEstimating optimal communities using {args.kappa_method}...")
        comm_sizes = compute_kappa(X, A, method=args.kappa_method, verbose=args.verbose)
        print(f"   Estimated: {comm_sizes}")
    elif args.comm_sizes:
        comm_sizes = args.comm_sizes
        print(f"\n   Using manual community sizes: {comm_sizes}")
    else:
        raise ValueError("Must either enable --compute_optimal_clusters or provide --comm_sizes")

    # Create model
    print(f"\nBuilding model...")
    model = HCD(
        nodes=X.shape[0],
        attrib=X.shape[1],
        method=args.method,
        ae_hidden_dims=args.ae_hidden_dims,
        comm_sizes=comm_sizes,
        ae_operator=args.ae_operator,
        ae_attn_heads=args.attn_heads,
        dropout=args.dropout,
        normalize_input=args.normalize_input,
        normalize_outputs=True,
        heads=1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Create trainer
    print(f"\nSetting up trainer...")
    trainer = Trainer(
        model=model,
        X=X,
        A=A,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_batch_learning=args.use_batch_learning,
        gamma=args.gamma,
        delta=args.delta,
        _lambda=args.lambda_,
        early_stopping=args.early_stopping,
        patience=args.patience,
        true_labels=true_labels,
        validation_data=validation_data,
        output_path=args.output_path,
        verbose=args.verbose
    )

    # Train model
    print(f"\nTraining model...")
    print("="*80)
    output = trainer.fit(device)

    # Display results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    print(f"\nTraining Summary:")
    print(f"   Total epochs: {len(output.train_loss_history)}")
    print(f"   Final training loss: {output.train_loss_history[-1]['Total Loss']:.4f}")

    if validation_data and output.validation_loss_history:
        print(f"   Final validation loss: {output.validation_loss_history[-1]['Total Loss']:.4f}")

    if output.performance_history:
        final_perf = output.performance_history[-1]
        print(f"\nFinal Performance Metrics:")

        metric_names = ['Homogeneity', 'Completeness', 'NMI', 'ARI']
        print(f"\n   Top Layer:")
        for name, value in zip(metric_names, final_perf[0][:4]):
            print(f"      {name:15s}: {value:.4f}")

        if len(final_perf) > 1 and final_perf[1] is not None:
            print(f"\n   Middle Layer:")
            for name, value in zip(metric_names, final_perf[1][:4]):
                print(f"      {name:15s}: {value:.4f}")

    print(f"\nPredicted Communities:")
    top_communities = len(torch.unique(output.predicted_train['top']))
    middle_communities = len(torch.unique(output.predicted_train['middle']))
    print(f"   Top layer: {top_communities} communities")
    print(f"   Middle layer: {middle_communities} communities")

    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.output_path, 'trained_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'comm_sizes': comm_sizes,
            'config': vars(args)
        }, model_path)
        print(f"\nModel saved to: {model_path}")

    print(f"\nResults saved to: {args.output_path}")
    print("\n" + "="*80)

    return output, model


if __name__ == "__main__":
    output, model = main()

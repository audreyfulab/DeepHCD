#!/usr/bin/env python3
"""
Inference Example: Load Trained Model and Predict on New Data

This example shows how to:
1. Load a previously trained model
2. Run inference on new data
3. Analyze predictions
"""

import torch
import argparse
from deephcd.model.model import HCD
from deephcd.utils.utilities import LoadData


def load_model(model_path, nodes, attrib):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')

    # Recreate model with saved configuration
    config = checkpoint['config']
    model = HCD(
        nodes=nodes,
        attrib=attrib,
        method=config['method'],
        ae_hidden_dims=config['ae_hidden_dims'],
        comm_sizes=checkpoint['comm_sizes'],
        ae_operator=config['ae_operator'],
        ae_attn_heads=config['attn_heads'],
        dropout=config['dropout'],
        normalize_input=config['normalize_input'],
        normalize_outputs=config.get('normalize_outputs', True),
        heads=1
    )

    # Load trained weights (use strict=False to allow minor mismatches)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    return model, checkpoint['comm_sizes']


def predict(model, X, A, device='cpu'):
    """Run inference on data."""
    model = model.to(device)
    X = X.to(device)
    A = A.to(device)

    with torch.no_grad():
        # Forward pass
        X_hat, A_hat, A_logit, X_all, A_all, P_all, S_all, AW = model.forward(X, A)

        # Get predictions
        top_predictions = S_all[0].cpu()
        middle_predictions = S_all[1].cpu()

        # Get probability distributions
        top_probs = P_all[0].cpu()
        middle_probs = [p.cpu() for p in P_all[1]]

    return {
        'top_labels': top_predictions,
        'middle_labels': middle_predictions,
        'top_probabilities': top_probs,
        'middle_probabilities': middle_probs,
        'reconstructed_features': X_hat.cpu(),
        'reconstructed_adjacency': A_hat.cpu()
    }


def main():
    
    parser = argparse.ArgumentParser(description='Run inference with trained HCD model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args([
    '--model_path', '/Users/jordandavis/Documents/DeepHCD_copy/DeepHCD/examples/training_output/trained_model.pth',
    '--data_path', '/Users/jordandavis/Documents/DeepHCD_copy/DeepHCD/examples/very_small_graph_150'
    ])


    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("="*70)
    print("DEEPHCD INFERENCE")
    print("="*70)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    pe, adj, idx_top, idx_mid, labels, sorted_top, sorted_mid = LoadData(args.data_path)

    # Prepare inputs
    X = torch.FloatTensor(pe[idx_mid, :])
    A = torch.FloatTensor(adj[:X.shape[0], :X.shape[0]]) + torch.eye(X.shape[0])
    nodes, features = X.shape

    print(f"   Data shape: {nodes} nodes, {features} features")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model, comm_sizes = load_model(args.model_path, nodes, features)
    print(f"   Loaded model with community sizes: {comm_sizes}")

    # Run inference
    print(f"\nRunning inference on {device}...")
    predictions = predict(model, X, A, device)

    # Display results
    print("\n" + "="*70)
    print("INFERENCE RESULTS")
    print("="*70)

    top_labels = predictions['top_labels']
    middle_labels = predictions['middle_labels']

    print(f"\nDetected Communities:")
    print(f"   Top layer: {len(torch.unique(top_labels))} communities")
    print(f"   Middle layer: {len(torch.unique(middle_labels))} communities")

    # Show distribution of nodes across communities
    print(f"\nCommunity Size Distribution:")
    print(f"   Top layer:")
    for comm in torch.unique(top_labels):
        count = (top_labels == comm).sum().item()
        print(f"      Community {comm.item()}: {count} nodes")

    print(f"\n   Middle layer:")
    for comm in torch.unique(middle_labels):
        count = (middle_labels == comm).sum().item()
        print(f"      Community {comm.item()}: {count} nodes")

    # Show confidence scores (average max probability per node)
    top_probs = predictions['top_probabilities']
    middle_probs = predictions['middle_probabilities']

    top_confidence = top_probs.max(dim=1)[0].mean().item()
    # Middle layer may have varying community sizes, compute mean carefully
    middle_confidence = torch.cat([p.max(dim=1)[0] for p in middle_probs]).mean().item()

    print(f"\nPrediction Confidence:")
    print(f"   Top layer average: {top_confidence:.4f}")
    print(f"   Middle layer average: {middle_confidence:.4f}")

    # Compare with true labels if available
    if sorted_top is not None:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score

        homogeneity_top = homogeneity_score(sorted_top, top_labels.numpy())
        completeness_top = completeness_score(sorted_top, top_labels.numpy())
        nmi_top = normalized_mutual_info_score(sorted_top, top_labels.numpy())
        ari_top = adjusted_rand_score(sorted_top, top_labels.numpy())

        print(f"\nComparison with True Labels (Top Layer):")
        print(f"   Homogeneity:  {homogeneity_top:.4f}")
        print(f"   Completeness: {completeness_top:.4f}")
        print(f"   NMI:          {nmi_top:.4f}")
        print(f"   ARI:          {ari_top:.4f}")

        if sorted_mid is not None:
            homogeneity_mid = homogeneity_score(sorted_mid, middle_labels.numpy())
            completeness_mid = completeness_score(sorted_mid, middle_labels.numpy())
            nmi_mid = normalized_mutual_info_score(sorted_mid, middle_labels.numpy())
            ari_mid = adjusted_rand_score(sorted_mid, middle_labels.numpy())

            print(f"\nComparison with True Labels (Middle Layer):")
            print(f"   Homogeneity:  {homogeneity_mid:.4f}")
            print(f"   Completeness: {completeness_mid:.4f}")
            print(f"   NMI:          {nmi_mid:.4f}")
            print(f"   ARI:          {ari_mid:.4f}")

    print("\n" + "="*70)

    return predictions


if __name__ == "__main__":
    predictions = main()

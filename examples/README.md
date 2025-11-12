# DeepHCD Examples

This directory contains comprehensive examples for using DeepHCD for hierarchical community detection in networks.

## Overview

DeepHCD is a deep learning framework for detecting hierarchical community structure in networks. These examples cover:

1. **Simulation**: Generate hierarchical networks with ground truth communities
2. **Training**: Train models to detect hierarchical communities
3. **Inference**: Use trained models to predict communities in new data

## Quick Start

### Complete Workflow (5 minutes)

```bash
# 1. Generate simulated data
python simple_simulation.py

# 2. Train a model
python simple_training.py

# Done! Model is trained and results are displayed
```

## Example Files

### Simulation Examples

| File | Description | Use When |
|------|-------------|----------|
| `simple_simulation.py` | Minimal simulation example | Quick testing, learning basics |
| `simulate_hierarchy_example.py` | Full-featured with CLI args | Custom networks, production use |
| `SIMULATION_README.md` | Complete simulation guide | Reference, advanced usage |

### Training Examples

| File | Description | Use When |
|------|-------------|----------|
| `simple_training.py` | Minimal training example | Quick start, learning |
| `train_hierarchy_example.py` | Full training with all options | Production, experimentation |
| `TRAINING_README.md` | Complete training guide | Reference, troubleshooting |

### Inference Examples

| File | Description | Use When |
|------|-------------|----------|
| `inference_example.py` | Load model and predict | Applying trained models |

### Full Example

| File | Description |
|------|-------------|
| `main.py` | End-to-end: simulate → train → evaluate |

## Documentation Structure

```
examples/
├── README.md (this file)              # Overview and quick start
├── SIMULATION_README.md               # Detailed simulation guide
├── TRAINING_README.md                 # Detailed training/inference guide
│
├── simple_simulation.py               # Quick simulation example
├── simulate_hierarchy_example.py      # Full simulation with options
│
├── simple_training.py                 # Quick training example
├── train_hierarchy_example.py         # Full training with options
├── inference_example.py               # Inference/prediction example
│
└── main.py                            # Complete end-to-end example
```

## Common Workflows

### Workflow 1: Quick Test

Test the package with minimal code:

```bash
python simple_simulation.py  # Creates ./my_simulated_network/
python simple_training.py     # Trains on the data
```

### Workflow 2: Custom Simulation + Training

Generate custom data and train:

```bash
# Generate custom network
python simulate_hierarchy_example.py \
    --top_layer_nodes 10 \
    --sample_size 1000 \
    --savepath ./custom_data/

# Train on it
python train_hierarchy_example.py \
    --data_path ./custom_data/ \
    --epochs 100 \
    --output_path ./custom_training/
```

### Workflow 3: Train and Deploy

Train, save, and use for inference:

```bash
# Train and save model
python train_hierarchy_example.py \
    --data_path ./my_data/ \
    --save_model \
    --output_path ./trained/

# Use for predictions
python inference_example.py \
    --model_path ./trained/trained_model.pth \
    --data_path ./new_data/
```

### Workflow 4: Full Pipeline

Complete analysis from scratch:

```bash
# Run the full example
python main.py

# This will:
# - Load/generate hierarchical network data
# - Estimate optimal community sizes
# - Train DeepHCD model
# - Evaluate performance
# - Save results
```

## Learning Path

### 1. Beginners

Start here to learn the basics:

1. Read this README
2. Run `simple_simulation.py` to see data generation
3. Run `simple_training.py` to see model training
4. Look at the code in these simple examples
5. Experiment with parameters

### 2. Intermediate Users

Ready for more control:

1. Read `SIMULATION_README.md` for simulation details
2. Use `simulate_hierarchy_example.py` with custom parameters
3. Read `TRAINING_README.md` for training details
4. Use `train_hierarchy_example.py` for full control
5. Experiment with different architectures and hyperparameters

### 3. Advanced Users

For production use and research:

1. Study `main.py` for complete workflow
2. Modify examples for your specific use case
3. Implement custom loss functions or architectures
4. Use `inference_example.py` as template for deployment
5. Refer to documentation for advanced features

## Parameter Quick Reference

### Simulation

```python
# Network structure
top_layer_nodes = 5          # Communities in top layer
nodes_per_super2 = (3, 3)    # Nodes per top community
nodes_per_super3 = (10, 12)  # Nodes per middle community

# Data generation
sample_size = 200            # Number of samples (observations)
SD = 0.1                     # Standard deviation for node values
```

### Training

```python
# Model architecture
ae_hidden_dims = [256, 128]  # Autoencoder hidden layers
ae_operator = 'GATv2Conv'    # Graph neural network type
comm_sizes = [15, 5]         # Communities [middle, top]

# Training
epochs = 50                  # Training iterations
learning_rate = 1e-3         # Learning rate
batch_size = 32              # Mini-batch size
```

## Tips and Best Practices

### Data Preparation

- Always normalize input features
- Add self-loops to adjacency matrix: `A + torch.eye(n)`
- Sort nodes by hierarchy for better visualization
- Check for disconnected components

### Model Training

- Start with automatic community size estimation
- Use early stopping to prevent overfitting
- Monitor both loss and evaluation metrics
- Try multiple random seeds for robustness
- Save successful model checkpoints

### Hyperparameter Tuning

- **Learning rate too high**: Loss oscillates or increases
  - Solution: Reduce to 1e-4 or 1e-5
- **Model too simple**: Poor performance even on training data
  - Solution: Increase `ae_hidden_dims` or add more layers
- **Model too complex**: Good training, poor test performance
  - Solution: Increase `dropout`, reduce model size
- **Wrong community sizes**: Poor clustering results
  - Solution: Try different `kappa_method` values

### Performance Optimization

- Use GPU when available: `device='cuda'`
- Enable batch learning for large graphs: `use_batch_learning=True`
- Reduce batch size if out of memory
- Use `GATv2Conv` for best accuracy, `SAGEConv` for speed

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `batch_size` to 16 or 8
- Reduce `ae_hidden_dims` sizes
- Use CPU: `device='cpu'`

**"Loss is NaN"**
- Reduce `learning_rate` to 1e-4
- Check for disconnected components in graph
- Normalize input features

**"Poor clustering performance"**
- Try different `kappa_method` for community estimation
- Adjust `delta` (modularity weight)
- Increase model capacity (`ae_hidden_dims`)
- Check that adjacency matrix has self-loops

**"Model not converging"**
- Reduce `learning_rate`
- Increase `batch_size`
- Adjust loss weights (`gamma`, `delta`, `lambda`)
- Enable early stopping

## Getting Help

1. **Check documentation**: Start with README files
2. **Review examples**: Look at similar use cases
3. **Examine code**: Examples are heavily commented
4. **Test with simple data**: Use `simple_*.py` examples first
5. **Check issues**: See GitHub issues for known problems

## Next Steps

After running these examples:

1. **Customize for your data**: Adapt examples to your use case
2. **Tune hyperparameters**: Experiment with settings
3. **Compare methods**: Try different GNN operators and methods
4. **Validate results**: Use domain knowledge to check communities
5. **Deploy models**: Use inference example as template

## Additional Resources

- Main package: `/deephcd/`
- Model implementation: `/deephcd/model/model.py`
- Training code: `/deephcd/model/train.py`
- Utilities: `/deephcd/utils/`
- Simulation code: `/deephcd/simulate/`

## Example Datasets

The examples use simulated data, but DeepHCD works with:

- Gene co-expression networks
- Social networks
- Citation networks
- Protein interaction networks
- Any network with node features and hierarchical structure

Adapt the data loading in examples to work with your format.

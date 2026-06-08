# Hierarchical Network Simulation Examples

This directory contains examples for simulating hierarchical networks with node-level data using DeepHCD.

## Quick Start

### Option 1: Minimal Example (Fastest)

Run the minimal simulation with default parameters:

```bash
python simple_simulation.py
```

This creates a small 3-layer hierarchy (~5 → 15 → 100 nodes) with 200 samples and saves to `./my_simulated_network/`.

### Option 2: Customizable Example

Run with command-line arguments for full control:

```bash
python simulate_hierarchy_example.py --top_layer_nodes 10 --sample_size 1000
```

Get help with all available options:

```bash
python simulate_hierarchy_example.py --help
```

## What Gets Generated

Each simulation creates:

### Network Files
- `top_layer_graph.pdf/png` - Top-level community structure
- `middle_layer_graph.pdf/png` - Middle-level sub-communities (3-layer only)
- `bottom_layer_graph.pdf/png` - Bottom-level individual nodes
- `directed_graphs.pkl` - Full network topology (Python pickle)
- `.npz` file - Adjacency matrices for all layers

### Data Files
- `_gexp.csv` - Pseudo-expression data (samples × nodes) in CSV format
- `_gexp.npy` - Same data in NumPy binary format
- `heatmaps.pdf/png` - Visualization of data correlations and network structure

### Metadata
- Node labels showing hierarchical community assignments
- Network statistics (modularity, degree distributions)

## Common Use Cases

### 1. Small Test Network

```bash
python simulate_hierarchy_example.py \
    --top_layer_nodes 3 \
    --nodes_per_super2 2 2 \
    --nodes_per_super3 5 8 \
    --sample_size 100 \
    --savepath ./small_test/
```

Creates: 3 → 6 → ~40 nodes with 100 samples

### 2. Medium Gene Network

```bash
python simulate_hierarchy_example.py \
    --top_layer_nodes 8 \
    --nodes_per_super2 3 5 \
    --nodes_per_super3 8 12 \
    --sample_size 500 \
    --SD 0.15 \
    --savepath ./gene_network/
```

Creates: 8 → ~32 → ~320 nodes with 500 samples (gene expression-like)

### 3. Large Sparse Network

```bash
python simulate_hierarchy_example.py \
    --top_layer_nodes 15 \
    --nodes_per_super2 4 6 \
    --nodes_per_super3 10 15 \
    --sample_size 1000 \
    --connect_prob_bottom 0.05 0.01 \
    --savepath ./large_sparse/
```

Creates: 15 → ~75 → ~900 nodes with sparse connections

### 4. Weighted Network

```bash
python simulate_hierarchy_example.py \
    --use_weighted_graph \
    --within_edgeweights 0.6 0.9 \
    --between_edgeweights 0.1 0.3 \
    --savepath ./weighted_network/
```

Generates weighted edges with specified ranges

### 5. Two-Layer Hierarchy

```bash
python simulate_hierarchy_example.py \
    --layers 2 \
    --top_layer_nodes 10 \
    --nodes_per_super2 15 20 \
    --savepath ./two_layer/
```

Creates a simpler 2-layer structure: 10 → ~175 nodes

## Loading Simulated Data

### Method 1: Using DeepHCD Utilities

```python
from deephcd.utils.utilities import LoadData

# Load everything
pe, adj, idx_top, idx_mid, labels, sorted_top, sorted_mid = LoadData('./my_simulated_network/')

print(f"Expression data: {pe.shape}")
print(f"Adjacency matrix: {adj.shape}")
```

### Method 2: Load Individual Files

```python
import numpy as np
import pandas as pd

# Load expression data
gexp = pd.read_csv('./my_simulated_network/_gexp.csv', index_col=0)

# Load NumPy data
pe = np.load('./my_simulated_network/_gexp.npy')

# Load network structure
data = np.load('./my_simulated_network/.npz', allow_pickle=True)
adj_layer1 = data['adj_layer1']
adj_layer2 = data['adj_layer2']
adj_layer3 = data['adj_layer3']  # For 3-layer networks
labels = data['labels']
```

## Key Parameters Explained

### Network Structure
- `--top_layer_nodes`: Number of top-level communities (e.g., tissue types)
- `--nodes_per_super2`: Min/max children per top node (creates middle layer)
- `--nodes_per_super3`: Min/max children per middle node (creates bottom layer)
- `--layers`: 2 or 3 hierarchical layers

### Topology
- `--subgraph_type`: 'small world', 'scale free', or 'random'
- `--subgraph_prob`: Edge probability within subgraphs [middle, bottom]
- `--connect_prob_middle/bottom`: [within, between] community connection probability

### Data Generation
- `--sample_size`: Number of samples (e.g., cells, individuals, time points)
- `--SD`: Standard deviation for simulated values
- `--common_dist`: If set, all parent nodes use N(0,σ); otherwise N(μₖ,σ)

### Graph Properties
- `--connect`: 'disc' (disconnected top) or 'full' (fully connected top)
- `--use_weighted_graph`: Generate weighted edges
- `--force_connect`: Ensure connectivity between linked communities
- `--mixed_graph`: Mix different subgraph topologies

## Understanding the Output

### Hierarchy Levels

```
Top Layer (5 nodes)
    ↓ (each spawns 2-4 middle nodes)
Middle Layer (~15 nodes)
    ↓ (each spawns 6-10 bottom nodes)
Bottom Layer (~120 nodes)
```

### Data Matrix Format

Expression data is organized as:

```
         Sample1  Sample2  ...  Sample500
Node1      0.45     0.32  ...     0.51
Node2      0.12     0.18  ...     0.09
...
Node120    0.73     0.81  ...     0.69
```

Where each node's values are simulated based on:
1. Its parent nodes in the DAG structure
2. The specified standard deviation
3. Community membership

### Network Adjacency

Each layer has an adjacency matrix showing connections:
- `adj_layer1`: Top-level community connections
- `adj_layer2`: Middle-level connections
- `adj_layer3`: Bottom-level (individual node) connections

## Tips

1. **Start small**: Test with small networks first to understand the structure
2. **Adjust sparsity**: Use `connect_prob` parameters to control edge density
3. **Reproducibility**: Always set `--set_seed` and `--seed_number` for reproducible results
4. **Memory**: Large networks (>1000 nodes) with many samples may require significant RAM
5. **Visualization**: Generated PDFs/PNGs help validate the network structure

## Example Workflow

```bash
# 1. Generate data
python simulate_hierarchy_example.py \
    --top_layer_nodes 5 \
    --sample_size 300 \
    --seed_number 123 \
    --savepath ./my_data/

# 2. Use in Python
python
>>> from deephcd.utils.utilities import LoadData
>>> pe, adj, *_ = LoadData('./my_data/')
>>> print(pe.shape)  # Check dimensions

# 3. Train model (see main.py for full example)
>>> from deephcd.model.model import HCD
>>> from deephcd.model.train import Trainer
>>> # ... training code ...
```

## Troubleshooting

**Problem**: "Too many nodes, simulation is slow"
- Reduce `top_layer_nodes`, `nodes_per_super2`, or `nodes_per_super3`
- Reduce `sample_size`

**Problem**: "Network is too sparse/dense"
- Adjust `connect_prob_middle` and `connect_prob_bottom`
- Change `subgraph_prob` values

**Problem**: "Can't load data"
- Check that `savepath` exists and contains the expected files
- Ensure you're using the correct path (with trailing slash)

## References

For more details on the simulation model and hierarchical network generation:
- See `deephcd/simulate/simulate.py` for the main simulation logic
- See `deephcd/simulate/graph.py` for network generation functions
- See `main.py` for a complete example including model training

# PaddleHelix API Reference

This document provides a complete reference of all public APIs in PaddleHelix, organized by module.

## Table of Contents

1. [Featurizer Module](#featurizer-module)
2. [Networks Module](#networks-module)
3. [Model Zoo Module](#model-zoo-module)
4. [Datasets Module](#datasets-module)
5. [Utils Module](#utils-module)
6. [Command Line Module](#command-line-module)

---

## Featurizer Module

### `pahelix.featurizer.featurizer`

#### `Featurizer`
Base class for all molecular featurizers.

**Methods:**
- `gen_features(raw_data: dict) -> dict | None`
- `collate_fn(batch_data_list: list) -> dict`

---

### `pahelix.featurizer.pretrain_gnn_featurizer`

#### `PreGNNAttrMaskFeaturizer`
Pre-training featurizer for attribute masking tasks.

**Constructor:**
```python
__init__(self, graph_wrapper, atom_type_num=None, mask_ratio=None)
```

**Parameters:**
- `graph_wrapper`: PGL GraphWrapper instance
- `atom_type_num` (int, optional): Total number of atom types
- `mask_ratio` (float, optional): Fraction of atoms to mask

**Methods:**
- `gen_features(raw_data: dict) -> dict | None`
- `collate_fn(batch_data_list: list) -> dict`

---

#### `PreGNNSupervisedFeaturizer`
Pre-training featurizer for supervised learning tasks.

**Constructor:**
```python
__init__(self, graph_wrapper)
```

**Parameters:**
- `graph_wrapper`: PGL GraphWrapper instance

**Methods:**
- `gen_features(raw_data: dict) -> dict | None`
- `collate_fn(batch_data_list: list) -> dict`

---

#### `PreGNNContextPredFeaturizer`
Pre-training featurizer for context prediction tasks.

**Constructor:**
```python
__init__(self, substruct_graph_wrapper, context_graph_wrapper, k, l1, l2)
```

**Parameters:**
- `substruct_graph_wrapper`: PGL GraphWrapper for substructure
- `context_graph_wrapper`: PGL GraphWrapper for context
- `k` (int): Substructure radius
- `l1` (int): Inner context boundary
- `l2` (int): Outer context boundary

**Methods:**
- `gen_features(raw_data: dict) -> dict | None`
- `collate_fn(batch_data_list: list) -> dict`

---

#### Utility Functions

**`reset_idxes(G: nx.Graph) -> tuple[nx.Graph, dict]`**
Reset node indices in NetworkX graph.

**`graph_data_obj_to_nx_simple(data: dict) -> nx.Graph`**
Convert graph data object to NetworkX graph.

**`nx_to_graph_data_obj_simple(G: nx.Graph) -> dict`**
Convert NetworkX graph to graph data object.

**`transform_contextpred(data: dict, k: int, l1: int, l2: int) -> tuple | None`**
Transform data for context prediction task.

---

## Networks Module

### `pahelix.networks.gnn_block`

#### Message Passing Functions

**`copy_send(src_feat: dict, dst_feat: dict, edge_feat: dict) -> tensor`**
Copy source features for message passing.

**`mean_recv(feat: tensor) -> tensor`**
Average pooling for message receiving.

**`sum_recv(feat: tensor) -> tensor`**
Sum pooling for message receiving.

**`max_recv(feat: tensor) -> tensor`**
Max pooling for message receiving.

**`unsqueeze(tensor: tensor) -> tensor`**
Add dimension to tensor.

---

#### GNN Layer Functions

**`gcn_layer(gw, feature, edge_features, act, name) -> tensor`**
Graph Convolutional Network layer.

**Parameters:**
- `gw`: PGL GraphWrapper
- `feature`: Node features tensor
- `edge_features`: Edge features tensor
- `act`: Activation function
- `name`: Layer name

---

**`gat_layer(gw, feature, edge_features, hidden_size, act, name, num_heads=1, feat_drop=0.1) -> tensor`**
Graph Attention Network layer.

**Parameters:**
- `gw`: PGL GraphWrapper
- `feature`: Node features tensor
- `edge_features`: Edge features tensor
- `hidden_size`: Hidden dimension size
- `act`: Activation function
- `name`: Layer name
- `num_heads`: Number of attention heads
- `feat_drop`: Feature dropout rate

---

**`gin_layer(gw, feature, edge_features, hidden_size, act, name) -> tensor`**
Graph Isomorphism Network layer.

**Parameters:**
- `gw`: PGL GraphWrapper
- `feature`: Node features tensor
- `edge_features`: Edge features tensor
- `hidden_size`: Hidden dimension size
- `act`: Activation function
- `name`: Layer name

---

### `pahelix.networks.lstm_block`

#### `LSTMBlock`
LSTM neural network block.

**Constructor:**
```python
__init__(self, input_size, hidden_size, num_layers=1, dropout=0.0)
```

**Parameters:**
- `input_size`: Input feature dimension
- `hidden_size`: Hidden state dimension
- `num_layers`: Number of LSTM layers
- `dropout`: Dropout probability

**Methods:**
- `forward(input_sequence) -> tuple[tensor, tuple]`

---

### `pahelix.networks.resnet_block`

#### `ResNetBlock`
Residual neural network block.

**Constructor:**
```python
__init__(self, input_dim, hidden_dim, output_dim, dropout=0.0)
```

**Parameters:**
- `input_dim`: Input dimension
- `hidden_dim`: Hidden layer dimension
- `output_dim`: Output dimension
- `dropout`: Dropout probability

**Methods:**
- `forward(input_tensor) -> tensor`

---

### `pahelix.networks.transformer_block`

#### `multi_head_attention`
Multi-head attention mechanism.

**Signature:**
```python
multi_head_attention(queries, keys, values, attn_bias, d_key, d_value, d_model, 
                    n_head=1, dropout_rate=0., cache=None, gather_idx=None, 
                    store=False, param_initializer=None, name="multi_head_att", 
                    is_test=False) -> tensor
```

**Parameters:**
- `queries`: Query tensor
- `keys`: Key tensor
- `values`: Value tensor
- `attn_bias`: Attention bias
- `d_key`: Key dimension
- `d_value`: Value dimension
- `d_model`: Model dimension
- `n_head`: Number of attention heads
- `dropout_rate`: Dropout probability
- `cache`: Cache for inference
- `gather_idx`: Gather index
- `store`: Whether to store attention weights
- `param_initializer`: Parameter initializer
- `name`: Layer name
- `is_test`: Whether in test mode

---

#### `positionwise_feed_forward`
Position-wise feed-forward network.

**Signature:**
```python
positionwise_feed_forward(x, d_inner_hid, d_hid, dropout_rate) -> tensor
```

**Parameters:**
- `x`: Input tensor
- `d_inner_hid`: Inner hidden dimension
- `d_hid`: Hidden dimension
- `dropout_rate`: Dropout probability

---

#### `pre_process_layer`
Pre-processing layer for transformer.

**Signature:**
```python
pre_process_layer(prev_out, out, dropout_rate) -> tensor
```

**Parameters:**
- `prev_out`: Previous output
- `out`: Current output
- `dropout_rate`: Dropout probability

---

#### `post_process_layer`
Post-processing layer for transformer.

**Signature:**
```python
post_process_layer(prev_out, out, dropout_rate) -> tensor
```

**Parameters:**
- `prev_out`: Previous output
- `out`: Current output
- `dropout_rate`: Dropout probability

---

### `pahelix.networks.optimizer`

#### `Adam`
Adam optimizer with learning rate scheduling.

**Constructor:**
```python
__init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, 
         parameter_list=None, weight_decay=None, grad_clip=None, name=None, 
         lazy_mode=False)
```

---

### `pahelix.networks.pre_post_process`

#### `pre_process_layer`
Pre-processing layer with residual connection and normalization.

**Signature:**
```python
pre_process_layer(prev_out, out, dropout_rate) -> tensor
```

#### `post_process_layer`
Post-processing layer with residual connection and normalization.

**Signature:**
```python
post_process_layer(prev_out, out, dropout_rate) -> tensor
```

---

## Model Zoo Module

### `pahelix.model_zoo.pretrain_gnns_model`

#### `PretrainGNNModel`
Pre-trained Graph Neural Network model.

**Constructor:**
```python
__init__(self, model_config={}, name='')
```

**Parameters:**
- `model_config` (dict): Model configuration dictionary
- `name` (str): Model name

**Configuration Options:**
- `hidden_size` (int): Hidden layer dimension (default: 256)
- `embed_dim` (int): Embedding dimension (default: 300)
- `dropout_rate` (float): Dropout probability (default: 0.5)
- `norm_type` (str): Normalization type (default: 'batch_norm')
- `graph_norm` (bool): Use graph normalization (default: False)
- `residual` (bool): Use residual connections (default: False)
- `layer_num` (int): Number of GNN layers (default: 5)
- `gnn_type` (str): GNN type - 'gin', 'gcn', 'gat' (default: 'gin')
- `JK` (str): Jumping Knowledge strategy (default: 'last')

**Methods:**
- `forward(graph_wrapper, is_test=False) -> tensor`

---

## Datasets Module

### `pahelix.datasets.inmemory_dataset`

#### `InMemoryDataset`
In-memory dataset for molecular data.

**Constructor:**
```python
__init__(self, data_list=None, npz_data_path=None)
```

**Parameters:**
- `data_list`: List of molecular data dictionaries
- `npz_data_path`: Path to cached NPZ data

**Methods:**
- `save_data(data_path: str) -> None`
- `__getitem__(key) -> InMemoryDataset | dict`
- `__len__() -> int`

---

### Dataset Classes

All dataset classes inherit from `InMemoryDataset` and provide task-specific data:

#### `BaceDataset`
β-secretase 1 inhibitor dataset.

#### `BBBPDataset`
Blood-brain barrier penetration dataset.

#### `ChemblFilteredDataset`
Filtered ChEMBL dataset.

#### `ClintoxDataset`
Clinical toxicity dataset.

#### `ESOLDataset`
Aqueous solubility dataset.

#### `FreeSolvDataset`
FreeSolv database dataset.

#### `HIVDataset`
HIV inhibitor dataset.

#### `LipophilicityDataset`
Lipophilicity dataset.

#### `MUVDataset`
Maximum Unbiased Validation dataset.

#### `SiderDataset`
Side Effect Resource dataset.

#### `Tox21Dataset`
Toxicology dataset with 12 tasks.

#### `ToxcastDataset`
Toxicology dataset.

#### `ZINCDataset`
ZINC database subset.

---

## Utils Module

### `pahelix.utils.compound_tools`

#### `CompoundConstants`
Constants for molecular features.

**Attributes:**
- `atom_num_list`: List of atomic numbers (1-118)
- `formal_charge_list`: List of formal charges (-5 to 5)
- `chiral_type_list`: List of chirality types
- `hybridization_type_list`: List of hybridization types
- `numH_list`: List of hydrogen counts (0-8)
- `implicit_valence_list`: List of implicit valences (0-6)
- `degree_list`: List of atom degrees (0-10)
- `bond_type_list`: List of bond types
- `bond_dir_list`: List of bond directions

---

#### Functions

**`mol_to_graph_data(mol) -> dict`**
Convert RDKit molecule to graph data.

**`smiles_to_graph_data(smiles) -> dict`**
Convert SMILES string to graph data.

**`get_atom_features(atom) -> list`**
Extract atom features from RDKit atom.

**`get_bond_features(bond) -> list`**
Extract bond features from RDKit bond.

---

### `pahelix.utils.splitters`

#### `Splitter`
Base class for dataset splitters.

**Methods:**
- `split(dataset, frac_train, frac_valid, frac_test) -> tuple`

---

#### `RandomSplitter`
Random dataset splitting.

**Methods:**
- `split(dataset, frac_train, frac_valid, frac_test, seed=None) -> tuple`

---

#### `IndexSplitter`
Sequential index-based splitting.

**Methods:**
- `split(dataset, frac_train, frac_valid, frac_test) -> tuple`

---

#### `ScaffoldSplitter`
Bemis-Murcko scaffold-based splitting.

**Methods:**
- `split(dataset, frac_train, frac_valid, frac_test, seed=None) -> tuple`

---

#### `RandomScaffoldSplitter`
Randomized scaffold-based splitting.

**Methods:**
- `split(dataset, frac_train, frac_valid, frac_test, seed=None) -> tuple`

---

#### Utility Functions

**`generate_scaffold(smiles, include_chirality=False) -> str`**
Generate Bemis-Murcko scaffold from SMILES.

---

### `pahelix.utils.data_utils`

#### Functions

**`save_data_list_to_npz(file_path, data_list) -> None`**
Save data list to NPZ format.

**`load_npz_to_data_list(file_path) -> list`**
Load data list from NPZ format.

---

### `pahelix.utils.paddle_utils`

#### Functions

**`load_model_state(model, model_path) -> None`**
Load model state from file.

**`save_model_state(model, model_path) -> None`**
Save model state to file.

---

## Command Line Module

### `pahelix.cmdline`

#### `main`
Main command line entry point.

**Signature:**
```python
main(args=None) -> Any
```

**Parameters:**
- `args`: Command line arguments (default: sys.argv[1:])

---

## Complete Import Guide

### Core Imports

```python
# Featurizers
from pahelix.featurizer import (
    Featurizer,
    PreGNNAttrMaskFeaturizer,
    PreGNNSupervisedFeaturizer,
    PreGNNContextPredFeaturizer
)

# Network blocks
from pahelix.networks.gnn_block import (
    gcn_layer, gat_layer, gin_layer,
    copy_send, mean_recv, sum_recv, max_recv
)

from pahelix.networks.lstm_block import LSTMBlock
from pahelix.networks.resnet_block import ResNetBlock
from pahelix.networks.transformer_block import (
    multi_head_attention,
    positionwise_feed_forward
)

# Models
from pahelix.model_zoo.pretrain_gnns_model import PretrainGNNModel

# Datasets
from pahelix.datasets import (
    InMemoryDataset,
    BaceDataset, BBBPDataset, ChemblFilteredDataset,
    ClintoxDataset, ESOLDataset, FreeSolvDataset,
    HIVDataset, LipophilicityDataset, MUVDataset,
    SiderDataset, Tox21Dataset, ToxcastDataset, ZINCDataset
)

# Utilities
from pahelix.utils.compound_tools import (
    CompoundConstants,
    mol_to_graph_data,
    smiles_to_graph_data
)

from pahelix.utils.splitters import (
    RandomSplitter, IndexSplitter, ScaffoldSplitter, RandomScaffoldSplitter
)

from pahelix.utils.data_utils import (
    save_data_list_to_npz,
    load_npz_to_data_list
)

from pahelix.utils.paddle_utils import (
    load_model_state,
    save_model_state
)
```

### Example Usage Patterns

```python
# 1. Load dataset
dataset = HIVDataset()

# 2. Split data
splitter = RandomSplitter()
train, valid, test = splitter.split(dataset, 0.8, 0.1, 0.1, seed=42)

# 3. Initialize featurizer
featurizer = PreGNNSupervisedFeaturizer(graph_wrapper)

# 4. Initialize model
model = PretrainGNNModel({
    'hidden_size': 256,
    'gnn_type': 'gin',
    'layer_num': 5
})

# 5. Process data
features = featurizer.collate_fn(batch_data)
output = model.forward(graph_wrapper)
```

This completes the comprehensive API reference for PaddleHelix. All public functions, classes, and methods are documented with their signatures, parameters, and return types.
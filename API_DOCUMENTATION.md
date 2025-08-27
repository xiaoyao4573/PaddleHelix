# PaddleHelix API Documentation

**PaddleHelix** (飞桨生物计算引擎) is a comprehensive bio-computing engine built on PaddlePaddle, designed for molecular property prediction, drug discovery, and computational chemistry applications.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Components](#core-components)
4. [Featurizers](#featurizers)
5. [Network Blocks](#network-blocks)
6. [Models](#models)
7. [Datasets](#datasets)
8. [Utilities](#utilities)
9. [Examples](#examples)
10. [Command Line Interface](#command-line-interface)

## Overview

PaddleHelix provides a complete toolkit for molecular machine learning, including:
- Molecular featurization and graph representation
- Graph Neural Network (GNN) architectures
- Pre-trained models for molecular property prediction
- Comprehensive molecular datasets
- Training and inference utilities

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pahelix

# Install dependencies
pip install -r requirements.txt

# Install PaddleHelix
python setup.py install
```

## Core Components

### Main Package Structure

```
pahelix/
├── featurizer/          # Molecular featurization
├── networks/            # Neural network architectures
├── model_zoo/          # Pre-trained models
├── datasets/            # Molecular datasets
├── utils/               # Utility functions
└── cmdline.py          # Command line interface
```

## Featurizers

The featurizer module provides tools to convert molecular structures (SMILES) into graph representations suitable for machine learning.

### Base Featurizer Class

```python
from pahelix.featurizer.featurizer import Featurizer

class Featurizer:
    """Base class for all molecular featurizers."""
    
    def gen_features(self, raw_data):
        """
        Generate features from raw molecular data.
        
        Args:
            raw_data (dict): Raw molecular data containing SMILES
            
        Returns:
            dict: Processed molecular features or None if failed
        """
        raise NotImplementedError()
    
    def collate_fn(self, batch_data_list):
        """
        Collate batch data for training.
        
        Args:
            batch_data_list (list): List of molecular data
            
        Returns:
            dict: Batched features ready for model input
        """
        raise NotImplementedError()
```

### PreGNN Attribute Mask Featurizer

```python
from pahelix.featurizer.pretrain_gnn_featurizer import PreGNNAttrMaskFeaturizer

# Initialize featurizer
featurizer = PreGNNAttrMaskFeaturizer(
    graph_wrapper=graph_wrapper,
    atom_type_num=119,  # Number of atom types + 2
    mask_ratio=0.15     # Ratio of atoms to mask
)

# Generate features for a single molecule
raw_data = {'smiles': 'CCO'}
features = featurizer.gen_features(raw_data)

# Process batch data
batch_data = [{'smiles': 'CCO'}, {'smiles': 'CCCO'}]
batch_features = featurizer.collate_fn(batch_data)
```

**Parameters:**
- `graph_wrapper`: PGL GraphWrapper for graph operations
- `atom_type_num`: Total number of atom types (default: 119 + 2)
- `mask_ratio`: Fraction of atoms to mask for self-supervised learning

**Features Generated:**
- `atom_type`: Atom type indices
- `chirality_tag`: Chirality information
- `bond_type`: Bond type indices
- `bond_direction`: Bond direction indices
- `masked_node_indice`: Indices of masked nodes
- `masked_node_label`: Original labels of masked nodes

### PreGNN Supervised Featurizer

```python
from pahelix.featurizer.pretrain_gnn_featurizer import PreGNNSupervisedFeaturizer

# Initialize featurizer
featurizer = PreGNNSupervisedFeaturizer(graph_wrapper=graph_wrapper)

# Generate features with labels
raw_data = {'smiles': 'CCO', 'label': 1.0}
features = featurizer.gen_features(raw_data)

# Process batch data
batch_data = [
    {'smiles': 'CCO', 'label': 1.0},
    {'smiles': 'CCCO', 'label': 0.0}
]
batch_features = featurizer.collate_fn(batch_data)
```

**Features Generated:**
- `supervised_label`: Normalized labels for supervised learning
- `valid`: Validity mask for labels

### PreGNN Context Prediction Featurizer

```python
from pahelix.featurizer.pretrain_gnn_featurizer import PreGNNContextPredFeaturizer

# Initialize featurizer
featurizer = PreGNNContextPredFeaturizer(
    substruct_graph_wrapper=substruct_wrapper,
    context_graph_wrapper=context_wrapper,
    k=3,    # Substructure radius
    l1=1,   # Inner context boundary
    l2=4    # Outer context boundary
)

# Generate features
raw_data = {'smiles': 'CCO'}
features = featurizer.gen_features(raw_data)
```

**Parameters:**
- `k`: Radius of substructure around root atom
- `l1`, `l2`: Context boundaries for context prediction

**Features Generated:**
- Substructure graph data
- Context graph data
- Overlap indices between substructure and context

### Utility Functions

```python
from pahelix.featurizer.pretrain_gnn_featurizer import (
    reset_idxes, 
    graph_data_obj_to_nx_simple, 
    nx_to_graph_data_obj_simple,
    transform_contextpred
)

# Reset node indices
G_new, mapping = reset_idxes(G)

# Convert to NetworkX format
nx_graph = graph_data_obj_to_nx_simple(graph_data)

# Convert from NetworkX format
graph_data = nx_to_graph_data_obj_simple(nx_graph)

# Transform for context prediction
substruct, context, assist = transform_contextpred(data, k=3, l1=1, l2=4)
```

## Network Blocks

The networks module provides building blocks for constructing neural network architectures.

### GNN Blocks

```python
from pahelix.networks.gnn_block import (
    gcn_layer, gat_layer, gin_layer,
    copy_send, mean_recv, sum_recv, max_recv
)

# Graph Convolutional Network layer
output = gcn_layer(
    gw=graph_wrapper,
    feature=node_features,
    edge_features=edge_features,
    act="relu",
    name="gcn_layer"
)

# Graph Attention Network layer
output = gat_layer(
    gw=graph_wrapper,
    feature=node_features,
    edge_features=edge_features,
    hidden_size=256,
    act="relu",
    name="gat_layer",
    num_heads=8,
    feat_drop=0.1
)

# Graph Isomorphism Network layer
output = gin_layer(
    gw=graph_wrapper,
    feature=node_features,
    edge_features=edge_features,
    hidden_size=256,
    act="relu",
    name="gin_layer"
)
```

**Parameters:**
- `gw`: PGL GraphWrapper
- `feature`: Node features tensor
- `edge_features`: Edge features tensor
- `act`: Activation function
- `name`: Layer name for parameter sharing
- `hidden_size`: Hidden dimension size
- `num_heads`: Number of attention heads (GAT)
- `feat_drop`: Feature dropout rate

### Message Passing Functions

```python
# Send function for message passing
def custom_send(src_feat, dst_feat, edge_feat):
    return src_feat["h"] + edge_feat["h"]

# Receive functions
mean_features = mean_recv(features)    # Average pooling
sum_features = sum_recv(features)      # Sum pooling
max_features = max_recv(features)      # Max pooling
```

### LSTM Block

```python
from pahelix.networks.lstm_block import LSTMBlock

lstm_block = LSTMBlock(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    dropout=0.1
)

output, (h_n, c_n) = lstm_block(input_sequence)
```

### ResNet Block

```python
from pahelix.networks.resnet_block import ResNetBlock

resnet_block = ResNetBlock(
    input_dim=128,
    hidden_dim=256,
    output_dim=128,
    dropout=0.1
)

output = resnet_block(input_tensor)
```

### Transformer Block

```python
from pahelix.networks.transformer_block import (
    multi_head_attention,
    positionwise_feed_forward,
    pre_process_layer,
    post_process_layer
)

# Multi-head attention
attention_output = multi_head_attention(
    queries=queries,
    keys=keys,
    values=values,
    attn_bias=attn_bias,
    d_key=64,
    d_value=64,
    d_model=256,
    n_head=8,
    dropout_rate=0.1
)

# Position-wise feed-forward network
ffn_output = positionwise_feed_forward(
    x=input_tensor,
    d_inner_hid=1024,
    d_hid=256,
    dropout_rate=0.1
)
```

## Models

### PretrainGNNModel

```python
from pahelix.model_zoo.pretrain_gnns_model import PretrainGNNModel

# Initialize model
model_config = {
    'hidden_size': 256,
    'embed_dim': 300,
    'dropout_rate': 0.5,
    'norm_type': 'batch_norm',
    'graph_norm': False,
    'residual': False,
    'layer_num': 5,
    'gnn_type': 'gin',  # 'gin', 'gcn', 'gat'
    'JK': 'last'        # 'last', 'sum', 'max', 'attention'
}

model = PretrainGNNModel(model_config=model_config, name='pretrain_gnn')

# Forward pass
output = model.forward(graph_wrapper, is_test=False)
```

**Model Configuration:**
- `hidden_size`: Hidden layer dimension
- `embed_dim`: Embedding dimension for atom/bond types
- `dropout_rate`: Dropout probability
- `norm_type`: Normalization type ('batch_norm', 'layer_norm')
- `graph_norm`: Whether to use graph normalization
- `residual`: Whether to use residual connections
- `layer_num`: Number of GNN layers
- `gnn_type`: Type of GNN ('gin', 'gcn', 'gat')
- `JK`: Jumping Knowledge strategy

**Supported GNN Types:**
- **GIN**: Graph Isomorphism Network
- **GCN**: Graph Convolutional Network  
- **GAT**: Graph Attention Network

## Datasets

PaddleHelix provides a comprehensive collection of molecular datasets for various tasks.

### Base Dataset Class

```python
from pahelix.datasets.inmemory_dataset import InMemoryDataset

# Create dataset from data list
dataset = InMemoryDataset(data_list=molecular_data)

# Create dataset from cached files
dataset = InMemoryDataset(npz_data_path='./cached_data/')

# Save dataset to disk
dataset.save_data('./cached_data/')

# Access data
molecule = dataset[0]           # Single molecule
subset = dataset[0:100]         # Slice of dataset
selected = dataset[[0, 5, 10]]  # Selected indices
```

### Available Datasets

```python
from pahelix.datasets import (
    BaceDataset, BBBPDataset, ChemblFilteredDataset,
    ClintoxDataset, ESOLDataset, FreeSolvDataset,
    HIVDataset, LipophilicityDataset, MUVDataset,
    SiderDataset, Tox21Dataset, ToxcastDataset, ZINCDataset
)

# Example: HIV dataset
hiv_dataset = HIVDataset()
print(f"Dataset size: {len(hiv_dataset)}")
print(f"Task type: {hiv_dataset.task_type}")
print(f"Metric: {hiv_dataset.metric}")

# Example: Tox21 dataset
tox21_dataset = Tox21Dataset()
print(f"Number of tasks: {tox21_dataset.num_tasks}")
```

**Dataset Properties:**
- **BACE**: β-secretase 1 inhibitor dataset
- **BBBP**: Blood-brain barrier penetration
- **ChEMBL**: Filtered ChEMBL dataset
- **Clintox**: Clinical toxicity dataset
- **ESOL**: Aqueous solubility dataset
- **FreeSolv**: FreeSolv database
- **HIV**: HIV inhibitor dataset
- **Lipophilicity**: Lipophilicity dataset
- **MUV**: Maximum Unbiased Validation dataset
- **SIDER**: Side Effect Resource dataset
- **Tox21**: Toxicology dataset
- **ToxCast**: Toxicology dataset
- **ZINC**: ZINC database subset

## Utilities

### Compound Tools

```python
from pahelix.utils.compound_tools import (
    CompoundConstants, mol_to_graph_data, smiles_to_graph_data
)

# Access compound constants
print(f"Atom types: {len(CompoundConstants.atom_num_list)}")
print(f"Bond types: {len(CompoundConstants.bond_type_list)}")
print(f"Chirality types: {len(CompoundConstants.chiral_type_list)}")

# Convert molecule to graph data
from rdkit.Chem import AllChem
mol = AllChem.MolFromSmiles('CCO')
graph_data = mol_to_graph_data(mol)

# Convert SMILES to graph data
graph_data = smiles_to_graph_data('CCO')
```

**Graph Data Structure:**
```python
{
    'atom_type': array([6, 6, 8]),           # Atom type indices
    'chirality_tag': array([0, 0, 0]),       # Chirality tags
    'edges': array([[0, 1], [1, 0], [1, 2], [2, 1]]),  # Edge indices
    'bond_type': array([1, 1, 1, 1]),        # Bond type indices
    'bond_direction': array([0, 0, 0, 0])    # Bond direction indices
}
```

### Data Splitters

```python
from pahelix.utils.splitters import (
    RandomSplitter, IndexSplitter, ScaffoldSplitter, RandomScaffoldSplitter
)

# Random splitting
random_splitter = RandomSplitter()
train, valid, test = random_splitter.split(
    dataset, 
    frac_train=0.8, 
    frac_valid=0.1, 
    frac_test=0.1,
    seed=42
)

# Scaffold-based splitting
scaffold_splitter = ScaffoldSplitter()
train, valid, test = scaffold_splitter.split(
    dataset,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42
)

# Random scaffold splitting
random_scaffold_splitter = RandomScaffoldSplitter()
train, valid, test = random_scaffold_splitter.split(
    dataset,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=42
)
```

**Splitting Strategies:**
- **RandomSplitter**: Random data splitting
- **IndexSplitter**: Sequential index-based splitting
- **ScaffoldSplitter**: Bemis-Murcko scaffold-based splitting
- **RandomScaffoldSplitter**: Randomized scaffold-based splitting

### Data Utilities

```python
from pahelix.utils.data_utils import (
    save_data_list_to_npz, load_npz_to_data_list
)

# Save data to NPZ format
save_data_list_to_npz('data.npz', molecular_data)

# Load data from NPZ format
loaded_data = load_npz_to_data_list('data.npz')
```

### Paddle Utilities

```python
from pahelix.utils.paddle_utils import (
    load_model_state, save_model_state
)

# Save model state
save_model_state(model, 'model_checkpoint.pdparams')

# Load model state
load_model_state(model, 'model_checkpoint.pdparams')
```

## Examples

### Complete Training Pipeline

```python
import paddle
from pahelix.datasets import HIVDataset
from pahelix.featurizer import PreGNNSupervisedFeaturizer
from pahelix.model_zoo import PretrainGNNModel
from pahelix.utils.splitters import RandomSplitter

# 1. Load dataset
dataset = HIVDataset()

# 2. Split data
splitter = RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.split(
    dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42
)

# 3. Initialize featurizer
featurizer = PreGNNSupervisedFeaturizer(graph_wrapper)

# 4. Initialize model
model_config = {
    'hidden_size': 256,
    'embed_dim': 300,
    'dropout_rate': 0.5,
    'layer_num': 5,
    'gnn_type': 'gin'
}
model = PretrainGNNModel(model_config=model_config)

# 5. Training loop
optimizer = paddle.optimizer.Adam(learning_rate=0.001)
for epoch in range(100):
    for batch_data in train_dataset:
        # Generate features
        features = featurizer.collate_fn(batch_data)
        
        # Forward pass
        output = model.forward(features)
        
        # Compute loss
        loss = compute_loss(output, features['supervised_label'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
```

### Pre-training with Attribute Masking

```python
from pahelix.featurizer import PreGNNAttrMaskFeaturizer

# Initialize attribute mask featurizer
featurizer = PreGNNAttrMaskFeaturizer(
    graph_wrapper=graph_wrapper,
    atom_type_num=119,
    mask_ratio=0.15
)

# Generate masked features
raw_data = {'smiles': 'CCO'}
features = featurizer.gen_features(raw_data)

# Process batch
batch_data = [{'smiles': 'CCO'}, {'smiles': 'CCCO'}]
batch_features = featurizer.collate_fn(batch_data)

# Access masked information
masked_indices = batch_features['masked_node_indice']
masked_labels = batch_features['masked_node_label']
```

### Context Prediction Pre-training

```python
from pahelix.featurizer import PreGNNContextPredFeaturizer

# Initialize context prediction featurizer
featurizer = PreGNNContextPredFeaturizer(
    substruct_graph_wrapper=substruct_wrapper,
    context_graph_wrapper=context_wrapper,
    k=3, l1=1, l2=4
)

# Generate context prediction features
raw_data = {'smiles': 'CCO'}
features = featurizer.gen_features(raw_data)

# Access substructure and context
substruct_data = features['transformed'][0]
context_data = features['transformed'][1]
assist_data = features['transformed'][2]
```

## Command Line Interface

PaddleHelix provides a command-line interface for common tasks.

```bash
# Run demo
python -m pahelix.cmdline

# Help
python -m pahelix.cmdline --help
```

## Advanced Usage

### Custom Featurizer

```python
from pahelix.featurizer.featurizer import Featurizer

class CustomFeaturizer(Featurizer):
    def __init__(self, custom_param):
        super().__init__()
        self.custom_param = custom_param
    
    def gen_features(self, raw_data):
        # Custom feature generation logic
        smiles = raw_data['smiles']
        # ... custom processing ...
        return processed_data
    
    def collate_fn(self, batch_data_list):
        # Custom batching logic
        # ... custom batching ...
        return batched_data
```

### Custom GNN Layer

```python
def custom_gnn_layer(gw, feature, edge_features, act, name):
    """Custom GNN layer implementation."""
    
    def custom_send(src_feat, dst_feat, edge_feat):
        return src_feat["h"] * edge_feat["h"]
    
    def custom_recv(feat):
        return layers.sequence_pool(feat, pool_type="sum")
    
    # Message passing
    msg = gw.send(custom_send,
            nfeat_list=[("h", feature)],
            efeat_list=[("h", edge_features)])
    
    output = gw.recv(msg, custom_recv)
    
    # Apply transformation
    output = layers.fc(output, size=feature.shape[-1], name=name)
    
    return output
```

### Model Ensemble

```python
class EnsembleModel:
    def __init__(self, models):
        self.models = models
    
    def forward(self, graph_wrapper, is_test=False):
        outputs = []
        for model in self.models:
            output = model.forward(graph_wrapper, is_test)
            outputs.append(output)
        
        # Ensemble strategy (e.g., average)
        ensemble_output = paddle.mean(paddle.stack(outputs), axis=0)
        return ensemble_output
```

## Performance Optimization

### Data Loading Optimization

```python
# Use multiple workers for data loading
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=featurizer.collate_fn
)
```

### Memory Management

```python
# Save memory by using gradient checkpointing
with paddle.amp.auto_cast():
    output = model.forward(graph_wrapper)
    loss = compute_loss(output, labels)

# Clear gradients efficiently
optimizer.clear_grad()
```

### Mixed Precision Training

```python
# Enable mixed precision training
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

with paddle.amp.auto_cast():
    output = model.forward(graph_wrapper)
    loss = compute_loss(output, labels)

scaled_loss = scaler.scale(loss)
scaled_loss.backward()
scaler.step(optimizer)
scaler.update()
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use gradient checkpointing
2. **Training Instability**: Adjust learning rate and use proper normalization
3. **Data Loading Errors**: Check SMILES validity and data format
4. **Model Convergence**: Use appropriate initialization and regularization

### Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data shapes
print(f"Node features shape: {node_features.shape}")
print(f"Edge features shape: {edge_features.shape}")

# Validate graph structure
print(f"Number of nodes: {graph_wrapper.num_nodes}")
print(f"Number of edges: {graph_wrapper.num_edges}")
```

## Contributing

To contribute to PaddleHelix:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 and Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## Citation

If you use PaddleHelix in your research, please cite:

```bibtex
@software{paddlehelix,
  title={PaddleHelix: A Comprehensive Bio-computing Engine},
  author={PaddlePaddle Team},
  year={2020},
  url={https://github.com/PaddlePaddle/PaddleHelix}
}
```

## Support

For questions and support:
- GitHub Issues: [Repository Issues](https://github.com/PaddlePaddle/PaddleHelix/issues)
- Documentation: [PaddleHelix Docs](https://paddlehelix.readthedocs.io/)
- Community: [PaddlePaddle Forum](https://discuss.paddlepaddle.org.cn/)
# PaddleHelix Quick Start Guide

This guide provides practical examples for common use cases in PaddleHelix, helping you get started quickly with molecular machine learning tasks.

## Table of Contents

1. [Installation](#installation)
2. [Basic Molecular Featurization](#basic-molecular-featurization)
3. [Working with Datasets](#working-with-datasets)
4. [Training a Simple GNN Model](#training-a-simple-gnn-model)
5. [Pre-training with Self-Supervised Learning](#pre-training-with-self-supervised-learning)
6. [Fine-tuning on Downstream Tasks](#fine-tuning-on-downstream-tasks)
7. [Model Inference](#model-inference)
8. [Advanced Examples](#advanced-examples)

## Installation

```bash
# Clone the repository
git clone https://github.com/PaddlePaddle/PaddleHelix.git
cd PaddleHelix

# Install dependencies
pip install paddlepaddle pgl rdkit numpy scikit-learn

# Install PaddleHelix
python setup.py install
```

## Basic Molecular Featurization

### Converting SMILES to Graph Representation

```python
from pahelix.utils.compound_tools import smiles_to_graph_data
from rdkit.Chem import AllChem

# Method 1: Direct SMILES conversion
smiles = "CCO"  # Ethanol
graph_data = smiles_to_graph_data(smiles)

print("Graph data structure:")
print(f"Number of atoms: {len(graph_data['atom_type'])}")
print(f"Number of bonds: {len(graph_data['bond_type'])}")
print(f"Atom types: {graph_data['atom_type']}")
print(f"Bond types: {graph_data['bond_type']}")

# Method 2: Using RDKit molecule
mol = AllChem.MolFromSmiles(smiles)
from pahelix.utils.compound_tools import mol_to_graph_data
graph_data = mol_to_graph_data(mol)
```

### Understanding Graph Data Structure

```python
# The graph_data contains:
{
    'atom_type': array([6, 6, 8]),           # C, C, O
    'chirality_tag': array([0, 0, 0]),       # No chirality
    'edges': array([[0, 1], [1, 0], [1, 2], [2, 1]]),  # Bidirectional edges
    'bond_type': array([1, 1, 1, 1]),        # Single bonds
    'bond_direction': array([0, 0, 0, 0])    # No direction
}
```

## Working with Datasets

### Loading and Exploring Datasets

```python
from pahelix.datasets import HIVDataset, Tox21Dataset

# Load HIV dataset
hiv_dataset = HIVDataset()
print(f"HIV dataset size: {len(hiv_dataset)}")
print(f"Task type: {hiv_dataset.task_type}")
print(f"Metric: {hiv_dataset.metric}")

# Examine first molecule
first_mol = hiv_dataset[0]
print(f"SMILES: {first_mol['smiles']}")
print(f"Label: {first_mol['label']}")

# Load Tox21 dataset (multi-task)
tox21_dataset = Tox21Dataset()
print(f"Tox21 dataset size: {len(tox21_dataset)}")
print(f"Number of tasks: {tox21_dataset.num_tasks}")
```

### Data Splitting Strategies

```python
from pahelix.utils.splitters import RandomSplitter, ScaffoldSplitter

# Random splitting
random_splitter = RandomSplitter()
train_data, valid_data, test_data = random_splitter.split(
    hiv_dataset, 
    frac_train=0.8, 
    frac_valid=0.1, 
    frac_test=0.1, 
    seed=42
)

print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

# Scaffold-based splitting (better for molecular data)
scaffold_splitter = ScaffoldSplitter()
train_data, valid_data, test_data = scaffold_splitter.split(
    hiv_dataset, 
    frac_train=0.8, 
    frac_valid=0.1, 
    frac_test=0.1, 
    seed=42
)
```

## Training a Simple GNN Model

### Setting Up the Training Pipeline

```python
import paddle
import paddle.fluid as fluid
from pahelix.featurizer import PreGNNSupervisedFeaturizer
from pahelix.model_zoo import PretrainGNNModel
from pahelix.networks.gnn_block import gin_layer
from pgl.graph_wrapper import GraphWrapper

# 1. Create a simple graph wrapper
def create_simple_graph_wrapper():
    # This is a simplified example - in practice you'd use PGL's GraphWrapper
    class SimpleGraphWrapper:
        def __init__(self):
            pass
        def to_feed(self, graph):
            return {
                'node_feat': graph.node_feat,
                'edge_feat': graph.edge_feat,
                'num_nodes': graph.num_nodes,
                'num_edges': graph.num_edges
            }
    return SimpleGraphWrapper()

# 2. Initialize featurizer
graph_wrapper = create_simple_graph_wrapper()
featurizer = PreGNNSupervisedFeaturizer(graph_wrapper)

# 3. Initialize model
model_config = {
    'hidden_size': 128,
    'embed_dim': 128,
    'dropout_rate': 0.3,
    'layer_num': 3,
    'gnn_type': 'gin'
}
model = PretrainGNNModel(model_config=model_config)

# 4. Simple training loop
optimizer = paddle.optimizer.Adam(learning_rate=0.001)

# Note: This is a conceptual example - actual implementation would require
# proper PGL GraphWrapper and data loading
```

## Pre-training with Self-Supervised Learning

### Attribute Masking Pre-training

```python
from pahelix.featurizer import PreGNNAttrMaskFeaturizer

# Initialize attribute mask featurizer
attr_mask_featurizer = PreGNNAttrMaskFeaturizer(
    graph_wrapper=graph_wrapper,
    atom_type_num=119,  # Number of atom types + 2
    mask_ratio=0.15     # Mask 15% of atoms
)

# Generate masked features for a molecule
raw_data = {'smiles': 'CCO'}
features = attr_mask_featurizer.gen_features(raw_data)

# Process batch data
batch_data = [
    {'smiles': 'CCO'},
    {'smiles': 'CCCO'},
    {'smiles': 'c1ccccc1'}  # Benzene
]
batch_features = attr_mask_featurizer.collate_fn(batch_data)

# Access masked information
masked_indices = batch_features['masked_node_indice']
masked_labels = batch_features['masked_node_label']

print(f"Masked {len(masked_indices)} atoms")
print(f"Masked indices: {masked_indices}")
print(f"Original labels: {masked_labels}")
```

### Context Prediction Pre-training

```python
from pahelix.featurizer import PreGNNContextPredFeaturizer

# Initialize context prediction featurizer
context_featurizer = PreGNNContextPredFeaturizer(
    substruct_graph_wrapper=graph_wrapper,
    context_graph_wrapper=graph_wrapper,
    k=3,    # Substructure radius
    l1=1,   # Inner context boundary
    l2=4    # Outer context boundary
)

# Generate context prediction features
raw_data = {'smiles': 'CCO'}
features = context_featurizer.gen_features(raw_data)

if features is not None:
    substruct_data, context_data, assist_data = features['transformed']
    print(f"Substructure atoms: {len(substruct_data['atom_type'])}")
    print(f"Context atoms: {len(context_data['atom_type'])}")
    print(f"Overlap indices: {assist_data['context_overlap_idx']}")
```

## Fine-tuning on Downstream Tasks

### Supervised Fine-tuning

```python
from pahelix.featurizer import PreGNNSupervisedFeaturizer

# Initialize supervised featurizer
supervised_featurizer = PreGNNSupervisedFeaturizer(graph_wrapper)

# Generate features with labels
raw_data = {'smiles': 'CCO', 'label': 1.0}
features = supervised_featurizer.gen_features(raw_data)

# Process batch data
batch_data = [
    {'smiles': 'CCO', 'label': 1.0},
    {'smiles': 'CCCO', 'label': 0.0},
    {'smiles': 'c1ccccc1', 'label': 0.0}
]
batch_features = supervised_featurizer.collate_fn(batch_data)

# Access labels
supervised_labels = batch_features['supervised_label']
valid_mask = batch_features['valid']

print(f"Labels: {supervised_labels}")
print(f"Valid mask: {valid_mask}")
```

## Model Inference

### Single Molecule Prediction

```python
def predict_single_molecule(model, featurizer, smiles):
    """Predict properties for a single molecule."""
    
    # Generate features
    raw_data = {'smiles': smiles}
    features = featurizer.gen_features(raw_data)
    
    if features is None:
        return None
    
    # Convert to graph wrapper format
    # Note: This is conceptual - actual implementation requires proper PGL setup
    graph_wrapper = convert_to_graph_wrapper(features)
    
    # Make prediction
    with paddle.no_grad():
        prediction = model.forward(graph_wrapper, is_test=True)
    
    return prediction

# Example usage
smiles_list = ['CCO', 'CCCO', 'c1ccccc1']
for smiles in smiles_list:
    prediction = predict_single_molecule(model, featurizer, smiles)
    if prediction is not None:
        print(f"{smiles}: {prediction}")
```

### Batch Prediction

```python
def predict_batch_molecules(model, featurizer, smiles_list, batch_size=32):
    """Predict properties for a batch of molecules."""
    
    predictions = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        
        # Prepare batch data
        batch_data = [{'smiles': s} for s in batch_smiles]
        
        # Generate batch features
        batch_features = featurizer.collate_fn(batch_data)
        
        # Convert to graph wrapper format
        graph_wrapper = convert_to_graph_wrapper(batch_features)
        
        # Make prediction
        with paddle.no_grad():
            batch_prediction = model.forward(graph_wrapper, is_test=True)
        
        predictions.extend(batch_prediction)
    
    return predictions
```

## Advanced Examples

### Custom Featurizer

```python
from pahelix.featurizer.featurizer import Featurizer

class CustomMorganFeaturizer(Featurizer):
    """Custom featurizer using Morgan fingerprints."""
    
    def __init__(self, radius=2, nBits=2048):
        super().__init__()
        self.radius = radius
        self.nBits = nBits
    
    def gen_features(self, raw_data):
        smiles = raw_data['smiles']
        mol = AllChem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        # Generate Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.nBits
        )
        
        # Convert to numpy array
        features = np.array(fp, dtype=np.float32)
        
        return {
            'morgan_fp': features,
            'smiles': smiles
        }
    
    def collate_fn(self, batch_data_list):
        # Stack Morgan fingerprints
        fps = [data['morgan_fp'] for data in batch_data_list]
        batch_fp = np.stack(fps, axis=0)
        
        return {
            'morgan_fp': batch_fp,
            'batch_size': len(batch_data_list)
        }

# Usage
custom_featurizer = CustomMorganFeaturizer(radius=3, nBits=1024)
features = custom_featurizer.gen_features({'smiles': 'CCO'})
```

### Model Ensemble

```python
class EnsembleGNNModel:
    """Ensemble of multiple GNN models."""
    
    def __init__(self, model_configs):
        self.models = []
        for config in model_configs:
            model = PretrainGNNModel(config)
            self.models.append(model)
    
    def forward(self, graph_wrapper, is_test=False):
        outputs = []
        
        for model in self.models:
            output = model.forward(graph_wrapper, is_test)
            outputs.append(output)
        
        # Ensemble strategy: average
        ensemble_output = paddle.mean(paddle.stack(outputs), axis=0)
        return ensemble_output

# Usage
model_configs = [
    {'hidden_size': 128, 'gnn_type': 'gin'},
    {'hidden_size': 256, 'gnn_type': 'gcn'},
    {'hidden_size': 128, 'gnn_type': 'gat'}
]

ensemble_model = EnsembleGNNModel(model_configs)
```

### Hyperparameter Tuning

```python
import itertools

def grid_search_hyperparameters():
    """Grid search for optimal hyperparameters."""
    
    # Define hyperparameter grid
    param_grid = {
        'hidden_size': [64, 128, 256],
        'layer_num': [3, 5, 7],
        'dropout_rate': [0.1, 0.3, 0.5],
        'gnn_type': ['gin', 'gcn', 'gat']
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    best_score = 0
    best_params = None
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        # Initialize model with current parameters
        model = PretrainGNNModel(model_config=params)
        
        # Train and evaluate (simplified)
        score = train_and_evaluate_model(model)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score

# Usage
best_params, best_score = grid_search_hyperparameters()
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

## Performance Optimization Tips

### Data Loading Optimization

```python
# Use multiple workers for data loading
from paddle.io import DataLoader

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
# Clear gradients efficiently
optimizer.clear_grad()

# Use gradient checkpointing for large models
with paddle.amp.auto_cast():
    output = model.forward(graph_wrapper)
    loss = compute_loss(output, labels)
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

## Troubleshooting Common Issues

### Memory Issues
- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision training

### Training Instability
- Adjust learning rate
- Use proper normalization
- Check data preprocessing

### Data Loading Errors
- Validate SMILES strings
- Check data format consistency
- Verify file paths

## Next Steps

1. **Explore the Tutorial**: Check out the Jupyter notebook in the `tutorial/` directory
2. **Run Examples**: Try the example scripts in the `apps/` directory
3. **Read Documentation**: Refer to the full API documentation
4. **Join Community**: Participate in discussions and ask questions

## Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive API reference
- **Examples**: Working code examples for common tasks
- **Community**: PaddlePaddle forum and discussions

This quick start guide should get you up and running with PaddleHelix quickly. Start with the basic examples and gradually explore more advanced features as you become comfortable with the framework.
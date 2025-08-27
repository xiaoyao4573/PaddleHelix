# PaddleHelix Documentation Index

Welcome to the comprehensive documentation for **PaddleHelix** (飞桨生物计算引擎), a powerful bio-computing engine built on PaddlePaddle for molecular machine learning and drug discovery applications.

## 📚 Documentation Overview

This repository contains complete documentation for PaddleHelix, organized into three main sections:

### 1. 📖 [Complete API Documentation](API_DOCUMENTATION.md)
**Comprehensive guide covering all aspects of PaddleHelix**

- **Overview & Installation**: Getting started with PaddleHelix
- **Core Components**: Detailed explanation of all modules
- **Featurizers**: Molecular featurization and graph representation
- **Network Blocks**: GNN architectures and building blocks
- **Models**: Pre-trained models and configurations
- **Datasets**: Comprehensive molecular datasets
- **Utilities**: Helper functions and tools
- **Examples**: Complete training pipelines and use cases
- **Advanced Usage**: Custom implementations and optimizations
- **Troubleshooting**: Common issues and solutions

**Best for**: Users who want to understand the complete framework and all available features.

### 2. 🔍 [API Reference](API_REFERENCE.md)
**Quick reference for all public APIs and functions**

- **Function Signatures**: Complete parameter lists and return types
- **Class Definitions**: All public classes and methods
- **Module Organization**: Clear structure by package
- **Import Guide**: Complete import statements
- **Usage Patterns**: Common code patterns and examples

**Best for**: Developers who need quick access to function signatures and API details.

### 3. 🚀 [Quick Start Guide](QUICKSTART_GUIDE.md)
**Practical examples for common use cases**

- **Installation**: Step-by-step setup instructions
- **Basic Examples**: Simple molecular featurization
- **Training Pipelines**: Complete model training examples
- **Pre-training**: Self-supervised learning examples
- **Fine-tuning**: Downstream task adaptation
- **Inference**: Model prediction and evaluation
- **Advanced Features**: Custom implementations
- **Performance Tips**: Optimization strategies

**Best for**: Users who want to get up and running quickly with practical examples.

## 🎯 Choose Your Path

### 🆕 **New to PaddleHelix?**
Start with the [Quick Start Guide](QUICKSTART_GUIDE.md) to get hands-on experience with basic examples.

### 🔧 **Building Applications?**
Use the [API Reference](API_REFERENCE.md) for quick access to function signatures and implementation details.

### 📚 **Want Complete Understanding?**
Read the [Complete API Documentation](API_DOCUMENTATION.md) for comprehensive coverage of all features and concepts.

## 🏗️ PaddleHelix Architecture

```
PaddleHelix
├── 🧬 Featurizers          # Molecular → Graph conversion
├── 🧠 Networks             # GNN architectures (GIN, GCN, GAT)
├── 🎯 Models               # Pre-trained models
├── 📊 Datasets             # Molecular datasets
├── 🛠️ Utilities            # Helper functions
└── 💻 CLI                  # Command line interface
```

## 🚀 Key Features

- **Molecular Featurization**: Convert SMILES to graph representations
- **Graph Neural Networks**: Support for GIN, GCN, and GAT architectures
- **Self-Supervised Learning**: Pre-training with attribute masking and context prediction
- **Comprehensive Datasets**: 13+ molecular datasets for various tasks
- **Flexible Architecture**: Easy to extend and customize
- **PaddlePaddle Integration**: Built on robust deep learning framework

## 📋 Quick Examples

### Basic Molecular Featurization
```python
from pahelix.utils.compound_tools import smiles_to_graph_data

# Convert SMILES to graph
smiles = "CCO"  # Ethanol
graph_data = smiles_to_graph_data(smiles)
print(f"Atoms: {len(graph_data['atom_type'])}")
```

### Load Dataset
```python
from pahelix.datasets import HIVDataset

dataset = HIVDataset()
print(f"Dataset size: {len(dataset)}")
```

### Initialize Model
```python
from pahelix.model_zoo import PretrainGNNModel

model = PretrainGNNModel({
    'hidden_size': 256,
    'gnn_type': 'gin',
    'layer_num': 5
})
```

## 🔗 Related Resources

- **GitHub Repository**: [PaddleHelix](https://github.com/PaddlePaddle/PaddleHelix)
- **PaddlePaddle**: [Main Framework](https://www.paddlepaddle.org.cn/)
- **PGL**: [Graph Learning Library](https://github.com/PaddlePaddle/PGL)
- **Tutorial**: Check the `tutorial/` directory for Jupyter notebooks
- **Examples**: Explore the `apps/` directory for complete applications

## 🤝 Contributing

We welcome contributions! Please see the contributing guidelines in the main repository.

## 📄 License

PaddleHelix is licensed under:
- Apache License 2.0
- Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

## 🆘 Need Help?

- **Documentation Issues**: Open an issue in this repository
- **Code Issues**: Use the main PaddleHelix repository
- **Questions**: Join the PaddlePaddle community discussions

---

**Happy Molecular Machine Learning with PaddleHelix! 🧬🧠**

*This documentation is maintained by the PaddleHelix community. For the latest updates, always check the main repository.*
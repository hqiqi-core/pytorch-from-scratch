# PyTorch from Scratch

Hands-on implementations for learning PyTorch and deep learning fundamentals.

This repository documents my learning process from basic tensor operations to building complete deep learning training pipelines with PyTorch.

## 🎯 Goals

The main goal is to develop a solid understanding of how modern deep learning models are implemented and trained, rather than relying only on high-level APIs.

The repository focuses on:

- PyTorch Tensor operations
- Automatic differentiation
- Neural network implementation
- Dataset and DataLoader
- Training and validation loops
- Optimization
- Learning rate scheduling
- Mixed-precision training
- Checkpointing
- GPU training
- Model evaluation

## 📚 Contents

```text
pytorch-from-scratch/
│
├── basics/
│   ├── tensor_operations.py
│   ├── autograd.py
│   └── device_management.py
│
├── nn/
│   ├── linear.py
│   ├── mlp.py
│   └── cnn.py
│
├── training/
│   ├── dataset.py
│   ├── trainer.py
│   ├── optimizer.py
│   └── scheduler.py
│
├── experiments/
│   └── ...
│
├── README.md
└── requirements.txt
```

The structure will evolve as new concepts are learned.

## 🧩 Learning Roadmap

### Phase 1 — Tensor & Autograd

-  Tensor creation and manipulation
-  Broadcasting
-  Matrix multiplication
-  Automatic differentiation
-  Computational graphs
-  `backward()`
-  `detach()`

### Phase 2 — Neural Networks

-  Linear layer
-  MLP
-  Activation functions
-  Loss functions
-  CNN
-  Normalization

### Phase 3 — Training Pipeline

-  Dataset
-  DataLoader
-  Training loop
-  Validation loop
-  Optimizer
-  Learning-rate scheduler
-  Checkpointing
-  Resume training

### Phase 4 — Efficient Training

-  GPU training
-  Automatic Mixed Precision
-  Gradient accumulation
-  Gradient clipping
-  Profiling

## 🔬 Experiments

Experiments will be added progressively as the implementation develops.

Each experiment will record:

- Configuration
- Dataset
- Model
- Training settings
- Evaluation metrics
- Observations
- Lessons learned

## 🚀 Philosophy

The purpose of this repository is not to reproduce tutorials line by line.

For important components, I will first understand the underlying mechanism and then implement it independently using PyTorch.

## 📌 Status

🚧 Under active development

This repository is part of my long-term preparation for multimodal learning and efficient VLM research.
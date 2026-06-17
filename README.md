# 🧩 Rubik-DL-Model

A modular Deep Learning framework for solving the **3×3 Rubik's Cube** using a stage-based approach inspired by the CFOP method.

Instead of training a single monolithic agent to solve the entire cube, this project decomposes the problem into four independent learning tasks:

- 🟡 Cross
- 🔵 First Two Layers (F2L)
- 🟢 Orientation of the Last Layer (OLL)
- 🔴 Permutation of the Last Layer (PLL)

Each stage has its own dataset generation process, training pipeline, neural network model and evaluation procedure.

---

# Motivation

Solving a Rubik's Cube is a highly combinatorial problem with an enormous search space.

Rather than relying exclusively on search algorithms or handcrafted heuristics, this project investigates whether Deep Learning models can learn the decision process of each solving stage individually.

The long-term goal is to create a modular solver where each stage can be improved independently and later integrated into a complete solving pipeline.

---

# Features

- Modular architecture
- Independent models for every CFOP stage
- Automatic dataset generation
- Neural-network based classifiers
- Checkpoint support
- Training and evaluation scripts
- Extensible architecture for future research

---

# Project Structure

```
Algorithms/
│
├── loader.py
├── OLL.csv
└── PLL.csv

Models/
│
├── classifier.py
├── base_model.py
├── astar.py
└── Checkpoints/

Stages/
│
├── base_stage.py
├── cross.py
├── f2l.py
├── oll.py
└── pll.py

Training/
│
├── generators.py
├── classifier_dataset.py
├── train_classifier.py
├── train_adi.py
├── evaluate.py
└── adi.py

train_cross.py
train_f2l.py
train_oll.py
train_pll.py
solver.py
cube_class.py
cube_representation.py
```

---

# Pipeline

```
Generate Cube States
        │
        ▼
Create Training Dataset
        │
        ▼
Train Stage Model
        │
        ▼
Evaluate
        │
        ▼
Save Checkpoint
        │
        ▼
Use inside Solver
```

---

# Current Status

| Module | Status |
|---------|--------|
| Cube Representation | ✅ |
| Cross | ✅ |
| F2L | ✅ |
| OLL | ✅ |
| PLL | ✅ |
| Dataset Generator | ✅ |
| Training Pipeline | ✅ |
| Evaluation | ✅ |
| Solver Integration | 🚧 |

---

# Research Goals

This repository is also intended as a research platform for exploring topics such as

- Curriculum Learning
- Representation Learning
- Reinforcement Learning
- Neural Search Heuristics
- Hybrid Search + Deep Learning
- Sample Efficiency
- Generalization to unseen cube states

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Training

Train one stage independently

```bash
python train_cross.py
```

or

```bash
python train_f2l.py
```

or

```bash
python train_oll.py
```

or

```bash
python train_pll.py
```

---

# Future Work

- End-to-end CFOP pipeline
- Transformer-based architectures
- Graph Neural Networks
- Reinforcement Learning fine-tuning
- Learned search heuristics
- Curriculum Learning
- Benchmark against classical solvers

---

# References

Agostinelli et al.
DeepCube

Kociemba Algorithm

CFOP Method

Sutton & Barto
Reinforcement Learning: An Introduction

---

# Author

Camilo Herrera

Universidad Industrial de Santander

Software Engineering

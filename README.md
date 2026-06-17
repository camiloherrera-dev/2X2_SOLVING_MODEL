# 🧩 Rubik-DL-Model

A modular Deep Learning framework for solving the **3×3 Rubik's Cube** using a stage-based approach inspired by the CFOP method.

Instead of training a single monolithic agent to solve the entire cube, this project decomposes the problem into four independent learning tasks:

- **Cross** — Solve the bottom cross
- **F2L** — Complete the first two layers
- **OLL** — Orient all pieces on the last layer
- **PLL** — Permute all pieces on the last layer to finish the cube

Each stage has its own dataset generator, neural network, training pipeline, and evaluation script.

---

## Architecture Overview

```
cube_class.py          ← Cube representation (3×3 numpy, all moves + rotation moves)
│
├── Stages/            ← One class per CFOP stage
│   ├── base_stages.py     BaseStage (ABC): value(), load/save weights
│   ├── cross.py           CrossStage
│   ├── f2l.py             F2lStage
│   ├── oll.py             OLLStage  (classifier + brute-force U search)
│   └── pll.py             PLLStage  (classifier + brute-force U search)
│
├── Models/            ← Neural network architectures
│   ├── base_model.py      StageModel  — value network (output: scalar)
│   ├── classifier.py      ClassifierModel — OLL/PLL classifier (output: class index)
│   └── astar.py           A* search guided by stage.value()
│
├── Training/          ← Data generation + training loops
│   ├── generators.py      State generators for each stage
│   ├── adi.py             ADIDataset (wraps generators for DataLoader)
│   ├── classifier_dataset.py  ClassifierDataset
│   ├── train_adi.py       Training loop for value networks (Cross, F2L)
│   ├── train_classifier.py    Training loop for classifiers (OLL, PLL)
│   └── evaluate.py        A*-based evaluator
│
├── Algorithms/        ← CSV tables of OLL/PLL algorithms
│   ├── OLL.csv
│   ├── PLL.csv
│   └── loader.py
│
├── train_cross.py     ← Entry point: train Cross stage
├── train_f2l.py       ← Entry point: train F2L stage (two-phase)
├── train_oll.py       ← Entry point: train OLL classifier
├── train_pll.py       ← Entry point: train PLL classifier
└── solver.py          ← Full pipeline: Cross → F2L → OLL → PLL
```

---

## Pipeline

```
Scrambled Cube
      │
      ▼
 CrossStage  ──► A* (value network heuristic) ──► Cross solved
      │
      ▼
 F2lStage    ──► A* (value network heuristic) ──► F2L solved
      │
      ▼
 OLLStage    ──► Classifier + U brute-force   ──► OLL solved
      │
      ▼
 PLLStage    ──► Classifier + U brute-force   ──► Cube solved ✅
```

---

## Installation

```bash
pip install torch numpy matplotlib
```

---

## Training

Train each stage independently:

```bash
python train_cross.py
python train_f2l.py
python train_oll.py
python train_pll.py
```

The F2L training runs in two phases automatically (easy scrambles first, then harder ones).

---

## Evaluation

```bash
python evaluator.py        # Cross stage evaluation
python test_oll.py         # OLL stage evaluation
python test_solver.py      # Full pipeline end-to-end
```

---

## Current Status

| Module | Status | Notes |
|---|---|---|
| Cube Representation | ✅ | 3×3 numpy, all 12 moves + M, x, y rotations |
| Cross | ✅ | Value network + A* |
| F2L | ✅ | Value network + A* (two-phase training) |
| OLL | ✅ | Classifier + U-adjustment brute force |
| PLL | ✅ | Classifier + U-adjustment brute force |
| Dataset Generators | ✅ | Per-stage, ADI-style |
| Training Pipeline | ✅ | ADI for value nets, CrossEntropy for classifiers |
| Evaluation | ✅ | A*-based, verbose step-by-step |
| Solver Integration | 🚧 | Pipeline functional but has bugs (see GUIDE.md) |

---

## Known Issues

See `GUIDE.md` for a detailed breakdown of bugs, their causes, and fixes.

Critical issues currently blocking `solver.py`:

- `is_oll_solved()` and `is_solved()` return method references instead of booleans
- `OLLStage.solve()` calls `self.is_solved()` which does not exist on the stage
- `astar.py` decodes faces as 2D arrays but cube stores them as numpy matrices (no bug here, but fragile)
- Case-sensitive path mismatches between `Algorithms/OLL.csv` and `algorithms/oll.csv`

---

## References

- Agostinelli et al. — *Solving the Rubik's Cube Without Human Knowledge* (DeepCubeA)
- Kociemba Algorithm
- CFOP Method
- Sutton & Barto — *Reinforcement Learning: An Introduction*

---

## Author

Camilo Herrera — Universidad Industrial de Santander, Software Engineering

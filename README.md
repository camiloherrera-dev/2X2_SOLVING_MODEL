# 🧩 2x2 Rubik's Cube Solver — Deep RL + A*

A reinforcement learning agent trained to solve the 2x2 Rubik's Cube, combining a neural network value function with A* search to find optimal or near-optimal solutions.

---

## 🧠 How It Works

The solver is inspired by the **DeepCube** approach (Agostinelli et al., 2019). Instead of hand-crafting heuristics, a neural network learns to estimate how far any cube state is from the solved state. This learned heuristic then guides an A* search to find the solution path.

**Pipeline:**
1. **Cube environment** (`cube_env.py`) — models the 2x2 state space and all valid moves
2. **State representation** (`cube_representation.py`) — encodes the cube as a numerical vector suitable for the network
3. **Neural network** (`DEEPCUBE.py`) — trained via self-play using Autodidactic Iteration (ADI) to predict cost-to-go
4. **Training loop** (`train_deepcube.py`) — generates scrambled cubes and trains the network iteratively
5. **Solver** (`solver.py`) — runs A* search guided by the trained network to find a solution
6. **Visualization** (`cube_net_print.py`) — prints the cube net to the terminal for debugging and demo

An alternative simpler approach using a **DQN (Deep Q-Network)** is also included in `/SIMPLEDQN` for comparison.

---

## 🗂️ Project Structure

```
├── cube_class.py          # Core cube data structure and move logic
├── cube_env.py            # RL environment (states, actions, rewards)
├── cube_representation.py # State encoding for the neural network
├── DEEPCUBE.py            # Neural network architecture
├── train_deepcube.py      # Training script (ADI)
├── solver.py              # A* search solver using the trained model
├── cube_net_print.py      # Terminal visualization of cube state
└── SIMPLEDQN/             # Alternative DQN-based approach
```

---

## 🚀 Getting Started

**Requirements:**
```bash
pip install torch numpy
```

**Train the model:**
```bash
python train_deepcube.py
```

**Run the solver:**
```bash
python solver.py
```

---

## 📚 References

- Agostinelli et al. (2019) — *Solving the Rubik's Cube Without Human Knowledge* ([paper](https://arxiv.org/abs/1805.07470))
- Sutton & Barto — *Reinforcement Learning: An Introduction*

---

## 👤 Author

**Camilo Herrera** — Software Engineering student @ Universidad Industrial de Santander  
[GitHub](https://github.com/Blinded4545)

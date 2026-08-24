```markdown
# Q-Learning GridWorld Simulation in Pure Python

A light-weight, zero-framework Reinforcement Learning (RL) environment built from scratch in Python. This project demonstrates how an autonomous agent uses **Q-Learning** to navigate a 6x6 grid environment, avoid wall collisions, and discover the optimal path to a designated goal.

---

##  Project Overview

Instead of relying on heavy frameworks like OpenAI Gymnasium or Stable-Baselines3, this project implements the fundamental mechanics of tabular Q-learning natively:

* **Custom Grid Environment:** A 6x6 discrete state space configured with custom start and goal coordinates.
* **Exploration vs. Exploitation:** Implements an $\epsilon$-greedy policy to balance random exploration ($\epsilon = 0.2$) during training with greedy exploitation ($\epsilon = 0.0$) during evaluation.
* **Tabular Q-Learning:** Implements the classic Bellman equation to iteratively update expected rewards across state-action pairs.
* **Heatmap Exporter:** Automatically exports a high-resolution Seaborn/Matplotlib heatmap displaying the maximum expected Q-value for every grid cell upon training completion.

---

## Reinforcement Learning Design

### State Space & Actions
* **States:** Discrete tuples representing the agent's current position $(r, c)$ on a 6x6 grid ($1 \le r, c \le 6$).
* **Actions:** `["up", "down", "left", "right"]`

### Reward Structure
* **Goal Reached:** $+10.0$ (Terminal state, ends episode)
* **Wall Collision:** $-2.0$ (Penalty step)
* **Standard Step:** $-0.1$ (Step cost encouraging shortest path finding)

---

##  Q-Table Heatmap Visualization

After training completes, the environment generates a heatmap visualizer showing max $Q(s, a)$ values for all positions:

![Q-Table Heatmap](grid_qtable_heatmap.png)

---

##  Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required visualization packages:

```bash
pip install numpy matplotlib seaborn

```

### Installation & Execution

1. **Clone the repository:**
```bash
git clone [https://github.com/orag214365-ship-it/qlearning-gridworld-python.git](https://github.com/orag214365-ship-it/qlearning-gridworld-python.git)
cd qlearning-gridworld-python

```


2. **Run the simulation:**
```bash
python main.py

```



---

##  Code Architecture

* `Agent`: Class managing state representations, action selection strategies, the underlying Q-table dictionary structure, training iterations, step execution, and heatmap rendering.
* `Agent.train()`: Executes training over $N$ episodes using Bellman equation updates.
* `Agent.export_qtable_heatmap()`: Parses internal $Q(s, a)$ dictionary entries into a matrix format and exports a clean visualization image via Matplotlib and Seaborn.

---

##  License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```

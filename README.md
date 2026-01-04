# Decentralized Heterogeneous ACO (Robotarium)

Decentralized, heterogeneous **Ant Colony Optimization (ACO)** for multi-robot coordination in the **Robotarium Python simulator**. Robots operate without a centralized planner, coordinating through local pheromone fields and limited neighbor communication. The framework supports heterogeneous agents, optional reinforcement learning, and constraint-aware control using quadratic programming.

---

## Overview

This project investigates **decentralized multi-agent coordination** using biologically inspired **Ant Colony Optimization (ACO)**. Each robot independently updates and follows pheromone gradients while exchanging limited local information with neighbors. The collective behavior enables coordinated foraging and navigation under communication and safety constraints.

The system is designed to be compatible with the **Robotarium execution model**, supports heterogeneous robot parameters, and optionally integrates **reinforcement learning (SAC)** for tuning coordination weights.

---

## Repository Structure

- `main.py` – Entry point for running the simulation  
- `environment.py` – Environment definition (home regions, food sources, pheromone maps)  
- `agent.py` – Per-robot logic (state machine, pheromone updates, neighbor interaction)  
- `controller.py` – High-level coordination, visualization, and RL hooks  
- `network_barriers.py` – QP-based network barrier certificates for connectivity and safety  
- `utils.py` – Global parameters and experiment configuration  
- `rl/` – Reinforcement learning utilities (SAC)
- `models/` – Saved RL checkpoints (optional)
- `environment.yml` – Conda environment specification

---

## Installation

### Option A: Conda (recommended)

NOTE: `environment.yml` in this repository is UTF-16 encoded.  
If Conda fails, open it once in VS Code and re-save it as UTF-8.

Run:

conda env create -f environment.yml  
conda activate robotariumVenv  

If needed, install remaining dependencies:

pip install stable-baselines3 cvxopt scipy matplotlib  

---

### Option B: Python venv

Create and activate a virtual environment:

python -m venv .venv  

Windows:
.venv\\Scripts\\activate  

macOS / Linux:
source .venv/bin/activate  

Install dependencies:

pip install -U pip  
pip install robotarium-python-simulator stable-baselines3 cvxopt scipy matplotlib numpy  

---

## Configuration

Most experiment parameters are defined in `utils.py`, including:
- Number of robots
- Environment bounds and timestep
- Communication radius
- Home and food regions
- Training flags for reinforcement learning

Modify these values and restart the simulation to apply changes.

---

## Running the Simulation

From the repository root:

python main.py  

For repeated automated runs:

python auto_runner.py  

---

## Reinforcement Learning (Optional)

The repository includes **Soft Actor-Critic (SAC)** utilities to learn or tune coordination parameters.

Relevant files:
- `rl/sac_agent.py`
- `rl/lambda_env.py`
- `rl/reward.py`
- `controller.py`

Enable training by adjusting flags in `utils.py`.

---

## Safety and Connectivity Constraints

Safety and network connectivity are enforced using **quadratic-programming-based barrier certificates** implemented in `network_barriers.py`. These constraints ensure collision avoidance and maintain communication links while preserving decentralized control.

---

## Troubleshooting

CVXOPT installation issues (Windows):

conda install -c conda-forge cvxopt  

Robotarium import errors:

pip show robotarium-python-simulator  

Blank plots or no visualization:
Ensure `matplotlib` is installed and you are not running in a headless environment.

---

## Acknowledgements

Built on the **Robotarium Python Simulator** and classical **Ant Colony Optimization** principles. Reinforcement learning components rely on **Stable-Baselines3 (SAC)**.

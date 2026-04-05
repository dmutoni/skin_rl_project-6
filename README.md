# DermRL — Reinforcement Learning for Skin Condition Management

**Student:** Denyse Mutoni Uwingeneye  
**Institution:** African Leadership University (ALU)  
**Assignment:** RL Summative — Comparing Value-Based and Policy Gradient Methods

---

## Problem Statement

Rwanda has only **13–14 dermatologists** serving nearly **14 million people**, leaving most patients — especially adolescents — with no access to dermatological care. This project trains RL agents to recommend daily lifestyle and treatment actions that clear skin conditions over a 90-day episode, forming the AI backbone of a mobile health app for Rwandan communities.

---

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py       # SkinConditionEnv — custom Gymnasium environment
│   └── rendering.py        # Pygame visualization
├── training/
│   ├── dqn_training.ipynb  # DQN — 10 hyperparameter experiments
│   ├── PPO_training.ipynb  # PPO — 10 hyperparameter experiments
│   └── reinforce_training.ipynb  # REINFORCE — 10 hyperparameter experiments
├── models/
│   ├── dqn/
│   │   └── best_model.zip  # Best DQN model (Large Buffer, +22.55 reward)
│   ├── ppo/
│   │   └── best_model.zip  # Best PPO model (Baseline, +21.81 reward)
│   └── reinforce/
│       └── best_model.pt   # Best REINFORCE model (Batch=8, +21.90 reward)
├── main.py                 # Run best agent with live visualization
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Environment — SkinConditionEnv

A custom Gymnasium environment simulating a **90-day skin condition treatment episode**.

| Component            | Details                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| Observation space    | `Box(7,)` — severity, inflammation, hydration, sun damage, stress, diet quality, day (normalised) |
| Action space         | `Discrete(8)` — Do Nothing, Exercise, Vitamins, Diet, Skincare, Pills, Topical, SPF               |
| Episode length       | 90 days (steps)                                                                                   |
| Start state          | Random patient — mild (sev 0.30–0.50), moderate (0.50–0.70), or severe (0.70–0.90)                |
| Terminal — recovery  | Composite severity < 0.35 → +20 reward bonus                                                      |
| Terminal — worsening | Composite severity ≥ 0.90 → −15 reward penalty                                                    |
| Truncation           | Day 90 reached → partial credit reward                                                            |

**Reward function:**

```
reward = (prev_severity - severity) × 10  - 0.1       # improvement + step cost
       + (hydration - 0.5) × 0.5                       # hydration bonus
       + (diet_quality - 0.5) × 0.5                    # diet bonus
       - stress × 0.3                                   # stress penalty
```

---

## Algorithms

| Algorithm | Type            | Library            | Best Reward   | Success Rate |
| --------- | --------------- | ------------------ | ------------- | ------------ |
| DQN       | Value-based     | Stable Baselines 3 | +22.55 ± 5.24 | 84%          |
| REINFORCE | Policy gradient | Custom PyTorch     | +21.90 ± 1.06 | 86.7%        |
| PPO       | Policy gradient | Stable Baselines 3 | +21.81 ± 0.86 | —            |

Each algorithm was trained with **10 different hyperparameter configurations**. See the training notebooks for full tables and analysis.

---

## Installation

```bash
git clone https://github.com/dmutoni/skin_rl_project-6.git
cd skin_rl_project-6
pip install -r requirements.txt
```

**requirements.txt:**

```
gymnasium==0.29.1
stable-baselines3==2.3.2
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pygame>=2.5.0
```

---

## Running the Best Agent

```bash
# Run best DQN agent — 3 episodes with pygame GUI (default)
python main.py

# Run 1 episode slowly (good for demos and video recording)
python main.py --episodes 1 --slow

# Run PPO instead
python main.py --algo ppo --episodes 3

# Run REINFORCE
python main.py --algo reinforce --episodes 3

# Terminal output only, no GUI
python main.py --no-render --episodes 5
```

**Example terminal output:**

```
==============================================================
  DermRL - Skin Condition Management via RL
  Algorithm : DQN
  Episodes  : 3
==============================================================

  Day   1 | Prescribed Pills             | Reward:  +1.23 | Severity: 0.681
  Day   2 | Topical Treatment            | Reward:  +0.94 | Severity: 0.612
  Day   3 | Improve Diet                 | Reward:  +0.61 | Severity: 0.574
  ...
  Episode 1 complete:
    Outcome      : RECOVERED
    Total reward : +23.41
    Steps taken  : 18 / 90 days
```

---

## Training Notebooks

Open any notebook in Google Colab and run all cells top to bottom. Each notebook:

- Writes `SkinConditionEnv` to `environment/custom_env.py`
- Trains 10 experiments with varied hyperparameters
- Saves all models to `models/`
- Produces reward curves, convergence plots, and generalization tests

---

## Key Results

- **DQN** achieves the highest peak reward (+22.55) with the Large Buffer (200k) configuration. Experience replay diversity is the critical factor for the 90-step episodes.
- **REINFORCE** reaches the highest success rate (86.7%) when batching 8 episodes per gradient update, which reduces Monte Carlo variance.
- **PPO** is the most stable — all 10 configurations converged within a narrow band (21.55–21.81), making it the safest choice for production deployment.

---

## Video Demo

## https://vimeo.com/1180329785/ea730b21e0

## License

MIT

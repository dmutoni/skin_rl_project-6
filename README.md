# DermRL 🧬 — Skin Condition Reinforcement Learning Environment

> An AI-driven skin condition simulation built as a Gymnasium environment,
> supporting DQN and PPO training for dermatological treatment planning.

---

## Project Vision

Millions of Rwandans — especially adolescents — lack access to dermatological care.
With only ~13 dermatologists serving nearly 14 million people, DermRL explores how
reinforcement learning can model, simulate, and optimise treatment pathways for
common skin conditions, forming the research backbone of a future AI-powered mobile
dermatology app.

---

## Folder Structure

```
skin_rl_project/
├── environment/
│   ├── custom_env.py        # Custom Gymnasium environment (SkinConditionEnv)
│   └── rendering.py         # OpenGL + Pygame visualisation dashboard
├── training/
│   ├── dqn_training.ipynb   # DQN training notebook (Google Colab)
│   └── pg_training.ipynb    # PPO training notebook (Google Colab)
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # Saved PPO models
├── main.py                  # Demo: random / trained agent with live visualisation
├── requirements.txt         # Project dependencies (MANDATORY)
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the demo (random agent + OpenGL window)
```bash
python main.py
python main.py --episodes 5          # run 5 episodes
python main.py --model dqn           # use trained DQN
python main.py --no-render           # headless / no window
```

### 3. Train in Google Colab
- Open `training/dqn_training.ipynb` in Colab → Runtime → T4 GPU
- Open `training/pg_training.ipynb` for PPO training
- Download `best_model.zip` and place in `models/dqn/` or `models/pg/`

---

## Environment Details

| Property | Value |
|---|---|
| Observation space | `Box(7,)` — severity, inflammation, hydration, sun_damage, stress, diet_quality, days_norm |
| Action space | `Discrete(8)` — 8 dermatologist-recommended actions |
| Episode length | 30 days (steps) |
| Success condition | Composite severity < 35% |
| Condition variants | Mild / Moderate / Severe (cycles across episodes) |

### Actions
| ID | Action | Effect |
|---|---|---|
| 0 | 💤 Do Nothing | Condition slowly worsens |
| 1 | 🏃 Exercise / Sports | Reduces stress, boosts circulation |
| 2 | 💊 Vitamin Supplements | Improves hydration & skin quality |
| 3 | 🥗 Improve Diet | Strongest diet quality boost |
| 4 | 🧴 Change Skincare Routine | Improves hydration, reduces inflammation |
| 5 | 💉 Prescribed Pills | Strongest severity reduction |
| 6 | 🧪 Topical Treatment | Moderate severity + inflammation reduction |
| 7 | 🕶️ Reduce Sun / Apply SPF | Reduces sun damage |

---

## Visualisation

The OpenGL dashboard (`rendering.py`) renders in real-time:
- **Procedural face model** — severity, inflammation, and hydration animate on the face
- **Severity timeline** — sparkline chart over 30 days
- **Health radar** — 6-axis radar chart for all metrics
- **Metric bars** — per-metric vertical progress bars
- **Action log** — scrolling history of decisions + reward deltas

---

## Tech Stack
- **Gymnasium** — RL environment API
- **Stable-Baselines3** — DQN + PPO implementations
- **PyTorch** — neural network backend
- **Pygame + PyOpenGL** — real-time OpenGL visualisation
- **Google Colab** — GPU-accelerated training

---

*Built for ALU coursework — AI for Dermatological Access in Rwanda*

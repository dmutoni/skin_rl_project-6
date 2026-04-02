"""
main.py — DermRL Demo
=====================
Runs a RANDOM agent in the SkinConditionEnv with full OpenGL visualization.

Usage:
    python main.py                        # random agent, human rendering
    python main.py --episodes 3           # run 3 episodes
    python main.py --model dqn            # load best saved DQN model
    python main.py --model ppo            # load best saved PPO model
    python main.py --no-render            # headless (no window)

Controls while window is open:
    ESC   — quit
"""

import argparse
import sys
import os
import time
import numpy as np
from typing import Optional

# ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(__file__))

from environment.custom_env import SkinConditionEnv, ACTION_LABELS, ACTION_ICONS, MAX_DAYS


def run_demo(episodes: int = 3, render: bool = True, model_type: Optional[str] = None):
    render_mode = "human" if render else "none"
    env = SkinConditionEnv(render_mode=render_mode, seed=0)

    # ── optionally load a trained model ──────────────────────────────────────
    agent = None
    if model_type is not None:
        try:
            if model_type.lower() == "dqn":
                from stable_baselines3 import DQN
                model_path = os.path.join("models", "dqn", "best_model.zip")
                agent = DQN.load(model_path)
                print(f"[INFO] Loaded DQN model from {model_path}")
            elif model_type.lower() in ("ppo", "pg"):
                from stable_baselines3 import PPO
                model_path = os.path.join("models", "pg", "best_model.zip")
                agent = PPO.load(model_path)
                print(f"[INFO] Loaded PPO model from {model_path}")
        except FileNotFoundError:
            print(f"[WARN] No saved model found at models/{model_type}/best_model.zip — falling back to random agent.")
            agent = None

    # ── episode loop ─────────────────────────────────────────────────────────
    all_rewards = []
    variant_names = ["Mild Acne", "Moderate Acne", "Severe Acne"]

    for ep in range(episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        done = False
        print(f"\n{'='*60}")
        print(f"  Episode {ep+1}/{episodes}  │  {variant_names[info['variant']]}  │  "
              f"Initial Severity: {info['severity']:.0%}")
        print(f"{'='*60}")
        print(f"  {'Day':>4}  │  {'Action':<28}  │  {'Severity':>9}  │  {'Reward':>8}")
        print(f"  {'-'*4}  │  {'-'*28}  │  {'-'*9}  │  {'-'*8}")

        while not done:
            # select action
            if agent is not None:
                action, _ = agent.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            icon  = ACTION_ICONS[action]
            label = ACTION_LABELS[action]
            print(f"  {info['day']:>4}  │  {icon} {label:<26}  │  "
                  f"{info['severity']:>8.1%}  │  {reward:>+8.3f}")

            if render:
                # small delay so animation is visible
                time.sleep(0.05)

        # episode summary
        final_sev = info["severity"]
        improved  = final_sev < 0.35
        print(f"\n  ── Episode {ep+1} Summary ──")
        print(f"  Total Reward  : {ep_reward:+.2f}")
        print(f"  Final Severity: {final_sev:.1%}")
        print(f"  Outcome       : {'✅ SKIN CLEARED' if improved else '⚠️  Needs More Treatment'}")
        all_rewards.append(ep_reward)

    # ── overall summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Overall Stats across {episodes} episodes")
    print(f"  Mean Reward : {np.mean(all_rewards):+.2f}")
    print(f"  Best Reward : {np.max(all_rewards):+.2f}")
    print(f"  Worst Reward: {np.min(all_rewards):+.2f}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermRL Demo")
    parser.add_argument("--episodes",  type=int,            default=3,
                        help="Number of episodes to run (default: 3)")
    parser.add_argument("--model",     type=str,            default=None,
                        choices=["dqn", "ppo", "pg", None],
                        help="Load a trained model: dqn | ppo (default: random agent)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable OpenGL rendering (headless mode)")
    args = parser.parse_args()

    run_demo(
        episodes   = args.episodes,
        render     = not args.no_render,
        model_type = args.model,
    )

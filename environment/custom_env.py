"""
SkinConditionEnv — Custom Gymnasium Environment
================================================
Simulates the progression/regression of a common skin condition (Acne Vulgaris)
over time in response to lifestyle and treatment actions recommended by dermatologists.

State space  : [severity, inflammation, hydration, sun_damage, stress, diet_quality, days_norm]
Action space : Discrete(8)
Episode length : 90 days (MAX_DAYS)
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List, Dict

MAX_DAYS        = 120
IMPROVEMENT_THR = 0.35
WORSENING_THR   = 0.90

ACTION_LABELS = [
    "Do Nothing", "Exercise / Sports", "Vitamin Supplements", "Improve Diet",
    "Change Skincare Routine", "Prescribed Pills", "Topical Treatment", "Reduce Sun / Apply SPF",
]
ACTION_ICONS = ["💤", "🏃", "💊", "🥗", "🧴", "💉", "🧪", "🕶️"]

ACTION_EFFECTS = {
    0: np.array([ 0.04,  0.02, -0.02,  0.01,  0.03, -0.02]),
    1: np.array([-0.03, -0.02, -0.01,  0.00, -0.08,  0.03]),
    2: np.array([-0.02, -0.01,  0.04,  0.00, -0.01,  0.02]),
    3: np.array([-0.04, -0.02,  0.02,  0.00, -0.01,  0.08]),
    4: np.array([-0.03, -0.03,  0.06,  0.00,  0.00,  0.01]),
    5: np.array([-0.10, -0.08,  0.01,  0.00, -0.01,  0.01]),
    6: np.array([-0.06, -0.05,  0.01,  0.00,  0.00,  0.00]),
    7: np.array([-0.01, -0.01,  0.02, -0.06,  0.00,  0.00]),
}
ACTION_NOISE = {0: 0.03, 1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.04, 6: 0.03, 7: 0.01}


class SkinConditionEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 4}

    def __init__(self, render_mode: str = "none", seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7, dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_LABELS))
        self._renderer = None
        self._state: np.ndarray = np.zeros(6, dtype=np.float64)
        self._day: int = 0
        self._episode_variant: int = 0
        self._history: List[Dict] = []

    def _get_obs(self) -> np.ndarray:
        return np.append(self._state, self._day / MAX_DAYS).astype(np.float32)

    def _get_info(self, action: int = -1) -> dict:
        return {
            "day": self._day, "action": action,
            "severity": float(self._state[0]), "inflammation": float(self._state[1]),
            "hydration": float(self._state[2]), "sun_damage": float(self._state[3]),
            "stress": float(self._state[4]), "diet_quality": float(self._state[5]),
            "variant": self._episode_variant,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # FIX 1: re-seed self._rng so reset(seed=N) gives reproducible episodes
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # FIX 2: derive variant from rng so it is reproducible per seed,
        # not tied to a global counter that drifts across eval/train envs
        self._episode_variant = int(self._rng.integers(0, 3))
        self._day = 0
        self._history = []
        if self._episode_variant == 0:
            self._state = self._rng.uniform([0.30,0.25,0.40,0.10,0.30,0.50],[0.50,0.40,0.60,0.25,0.50,0.70])
        elif self._episode_variant == 1:
            self._state = self._rng.uniform([0.50,0.45,0.25,0.20,0.50,0.30],[0.70,0.60,0.40,0.40,0.70,0.50])
        else:
            self._state = self._rng.uniform([0.70,0.65,0.10,0.30,0.65,0.15],[0.90,0.85,0.25,0.55,0.85,0.35])
        self._history.append(self._get_info())
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not valid. Expected int in [0, {self.action_space.n - 1}].")
        self._day += 1
        delta = ACTION_EFFECTS[action].copy()
        noise = self._rng.normal(0, ACTION_NOISE[action], size=6)
        self._state = np.clip(self._state + delta + noise, 0.0, 1.0)
        severity = self._state[0]
        inflammation = self._state[1]
        composite = 0.6 * severity + 0.3 * inflammation + 0.1 * self._state[3]
        prev_severity = self._history[-1]["severity"] if self._history else severity
        improvement = prev_severity - severity
        reward = improvement * 10.0 - 0.1
        reward += (self._state[2] - 0.5) * 0.5
        reward += (self._state[5] - 0.5) * 0.5
        reward -= self._state[4] * 0.3
        terminated = truncated = False
        if composite < IMPROVEMENT_THR:
            reward += 20.0; terminated = True
        elif composite >= WORSENING_THR:
            reward -= 15.0; terminated = True
        elif self._day >= MAX_DAYS:
            reward += max(0, (0.7 - severity) * 10); truncated = True
        self._history.append(self._get_info(action))
        if self.render_mode == "human": self.render()
        return self._get_obs(), float(reward), terminated, truncated, self._get_info(action)

    def render(self):
        if self.render_mode == "none": return
        if self._renderer is None:
            from environment.rendering import SkinRenderer
            self._renderer = SkinRenderer()
        return self._renderer.render(self._history, self._day, self._episode_variant)

    def close(self):
        if self._renderer is not None:
            self._renderer.close(); self._renderer = None

    @property
    def history(self): return list(self._history)
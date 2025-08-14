# rl_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PairsTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df, window_size=30, commission=0.001):
        super(PairsTradingEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.commission = commission
        
        # Actions: 0 = Hold/Exit, 1 = Long Spread, 2 = Short Spread
        self.action_space = spaces.Discrete(3)
        
        # Observations: [z_score, position, volatility, momentum]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -1.0, 0.0, -0.5]),
            high=np.array([5.0, 1.0, 1.0, 0.5]),
            dtype=np.float32
        )
        
        self._prepare_data()

    def _prepare_data(self):
        """Pre-calculates all features for the agent."""
        self.df['rolling_mean'] = self.df['spread'].rolling(window=self.window_size).mean()
        self.df['rolling_std'] = self.df['spread'].rolling(window=self.window_size).std()
        self.df['z_score'] = (self.df['spread'] - self.df['rolling_mean']) / self.df['rolling_std']
        self.df['spread_returns'] = self.df['spread'].pct_change().fillna(0)
        self.df['volatility'] = self.df['spread_returns'].rolling(window=self.window_size).std()
        self.df['momentum'] = self.df['spread'].pct_change(periods=5).fillna(0)
        self.df.dropna(inplace=True)
        self.start_tick = 0
        self.end_tick = len(self.df) - 1

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_tick = self.start_tick
        self.position = 0
        self.cumulative_reward = 0
        self.trades = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        """Returns the current state of the environment, clipping values to fit the space."""
        z_score = np.clip(self.df['z_score'].iloc[self.current_tick], -5.0, 5.0)
        volatility = self.df['volatility'].iloc[self.current_tick]
        momentum = np.clip(self.df['momentum'].iloc[self.current_tick], -0.5, 0.5)
        
        return np.array([z_score, self.position, volatility, momentum], dtype=np.float32)

    def step(self, action):
        self.current_tick += 1
        terminated = self.current_tick >= self.end_tick

        reward = self.df['spread_returns'].iloc[self.current_tick] * self.position
        
        prev_position = self.position
        
        if action == 0: self.position = 0
        elif action == 1: self.position = 1
        elif action == 2: self.position = -1
            
        if self.position != prev_position:
            self.trades += 1
            reward -= self.commission
            
        self.cumulative_reward += reward
        observation = self._get_observation()
        info = {'cumulative_reward': self.cumulative_reward, 'trades': self.trades}
        truncated = False

        return observation, reward, terminated, truncated, info
import gym
import numpy as np
from utils import MAX_TIME

class LambdaEnv(gym.Env):
    """
    Custom Environment for Lambda control using SAC.
    Lambda is the heterogeneous weight to be used in the connectivity vs. exploration QP: min (lambda * ConnectivityCost + (1-lambda) * ExplorationCost )
    The action is continuous: the lambda value [0, 1].
    """
    def __init__(self, get_state_fn, compute_reward_fn):
        super(LambdaEnv, self).__init__()

        self.get_state_fn = get_state_fn
        self.compute_reward_fn = compute_reward_fn

        # Dynamically determine state dimension
        initial_state = np.array(self.get_state_fn(), dtype=np.float32)
        state_dim = initial_state.shape[0]

        # Action space: lambda value (continuous between 0 and 1)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: dynamically sized based on state vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        self.state = initial_state

        self.prev_lambda = None


    def reset(self):
        # Reset environment state
        self.state = self.get_state_fn()
        return np.array(self.state, dtype=np.float32)


    def step(self, action, current_time):
        lambda_value = float(action[0])  # Convert action to scalar
        alpha = 0.4  # smoothing factor, smaller = smoother
        if self.prev_lambda is not None:
            lambda_value = (1 - alpha) * self.prev_lambda + alpha * lambda_value
        lambda_value = np.clip(lambda_value, 0.05, 0.95)


        # Update state and compute reward
        next_state = self.get_state_fn()
        reward = self.compute_reward_fn(next_state, lambda_value, self.prev_lambda)

        done = current_time >= MAX_TIME  # Episode end condition, optional

        self.state = next_state
        self.prev_lambda = lambda_value

        return np.array(next_state, dtype=np.float32), reward, done, {}
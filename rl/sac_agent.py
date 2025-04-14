import os
from stable_baselines3 import SAC
from .lambda_env import LambdaEnv
from stable_baselines3.common.logger import configure
import torch


class AgentSAC:
    def __init__(self, agent_index, get_state_fn, compute_reward_fn, logger, training_interval=10):
        self.agent_index = agent_index
        self.training_interval = training_interval
        self.training_step = 0
        self.logger = logger
        self.episode_reward = 0.0

        self.env = LambdaEnv(get_state_fn, compute_reward_fn)
        self.model = SAC(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_rate=1e-4,  # Smaller learning rate for smoother updates
            buffer_size=100_000,  # Large buffer
            batch_size=512,  # Larger batches for stability
            tau=0.005,  # Target network smoothing coefficient
            gamma=0.99,  # Discount factor
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto_0.05",  # << Lower entropy coefficient to reduce exploration noise
            target_entropy=-0.1,   # << This controls how random the action is: lower = smoother!
            use_sde=True,  # State-dependent exploration for smoother action noise
            sde_sample_freq=4,  # How often to resample noise (larger = smoother)

            device="cuda" if torch.cuda.is_available() else "cpu",  # Optional: for performance
        )
        self.model._logger = configure()

        # Optional: load existing model
        model_path = f'models/agent_{agent_index}_lambda_sac.zip'
        if os.path.exists(model_path):
            self.model = SAC.load(model_path, env=self.env)
            self.model._logger = configure()


            buffer_path = f'models/agent_{self.agent_index}_replay_buffer.pkl'
            if os.path.exists(buffer_path):
                self.model.load_replay_buffer(buffer_path)

    def select_lambda(self, current_time):
        obs = self.env.state
        action, _ = self.model.predict(obs, deterministic=False)
        lambda_value = float(action[0])

        # Step environment
        next_obs, reward, done, _ = self.env.step(action, current_time)

        self.episode_reward += reward

        # Log lambda and reward
        self.logger.writerow([current_time, lambda_value, reward, self.episode_reward])


        # Add to replay buffer
        # print(f"State vector shape: {obs.shape}, Reward: {reward}")
        self.model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])

        # Train every N steps
        self.training_step += 1
        if self.training_step % self.training_interval == 0:
            self.model.train(batch_size=256, gradient_steps=1)

        return lambda_value

    def save(self):
        self.model.save(f'models/agent_{self.agent_index}_lambda_sac')
        if self.model.replay_buffer is not None:
            self.model.save_replay_buffer(f'models/agent_{self.agent_index}_replay_buffer.pkl')

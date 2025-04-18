import os
from stable_baselines3 import SAC
from .lambda_env import LambdaEnv
from stable_baselines3.common.logger import configure


class AgentSAC:
    def __init__(self, agent_index, get_state_fn, compute_reward_fn, logger, training_interval=10):
        self.agent_index = agent_index
        self.training_interval = training_interval
        self.training_step = 0
        self.logger = logger
        self.episode_reward = 0.0

        self.env = LambdaEnv(get_state_fn, compute_reward_fn)
        self.model = SAC("MlpPolicy", self.env, verbose=0, learning_rate=1e-4, buffer_size=10000000)
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

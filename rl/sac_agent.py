import os
from stable_baselines3 import SAC
from .lambda_env import LambdaEnv

class AgentSAC:
    def __init__(self, agent_index, get_state_fn, apply_lambda_fn, compute_reward_fn, logger, training_interval=10):
        self.agent_index = agent_index
        self.training_interval = training_interval
        self.training_step = 0
        self.logger = logger
        self.episode_reward = 0.0

        self.env = LambdaEnv(get_state_fn, apply_lambda_fn, compute_reward_fn)
        self.model = SAC("MlpPolicy", self.env, verbose=0)

        # Optional: load existing model
        model_path = f'models/agent_{agent_index}_lambda_sac.zip'
        if os.path.exists(model_path):
            self.model = SAC.load(model_path, env=self.env)

    def select_lambda(self, current_time):
        obs = self.env.state
        action, _ = self.model.predict(obs, deterministic=False)
        lambda_value = float(action[0])

        # Step environment
        next_obs, reward, done, _ = self.env.step(action)

        self.episode_reward += reward

        # Log lambda and reward
        if done:
            self.logger.writerow(["EPISODE_END", "", "", self.episode_reward])
            self.episode_reward = 0.0
        else:
            self.logger.writerow([current_time, lambda_value, reward, ""])

        # Add to replay buffer
        self.model.replay_buffer.add(obs, action, reward, next_obs, done)

        # Train every N steps
        self.training_step += 1
        if self.training_step % self.training_interval == 0:
            self.model.train(batch_size=256, gradient_steps=1)

        return lambda_value

    def save(self):
        self.model.save(f'models/agent_{self.agent_index}_lambda_sac')

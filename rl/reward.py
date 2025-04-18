# We can design the reward as we wish, depending on designed state vector

from environment import *
import numpy as np

def compute_reward(state_vector, lambda_value):

    local_density, x, y, vx, vy, is_returning, progress, num_pheromones = state_vector

    # Penalize disconnection hard
    disconnection_penalty = -3.0 if local_density == 0.0 else 0.0 

    # Reward 1: trail-following encouragement when sparse
    follow_gain = 1.0
    reward_follow = (1 - local_density) * lambda_value * follow_gain

    # Reward 2: avoidance encouragement when crowded
    avoid_gain = 1.5
    reward_avoid = local_density * (1 - lambda_value) * avoid_gain

    # Reward 3: smooth motion or velocity alignment reward
    # reward_alignment = velocity_alignment * lambda_value * 0.1

    # Reward 4: finding food or getting back home, cumulative
    progress_scale = 3.0 # tunable
    reward_progress = np.log1p(progress * progress_scale)

    # Penalize stagnation
    speed = np.sqrt(vx**2 + vy**2)  
    min_speed_threshold = 0.15
    stagnation_penalty_scale = 2.0
    stagnation_penalty = - stagnation_penalty_scale * max(0.0, (min_speed_threshold - speed))

    # Encourage exploration if few pheromones detected
    exploration_gain = 0.8
    reward_exploration = (1 - num_pheromones) * (1 - lambda_value) * exploration_gain



    # Final reward (weight terms as needed)
    total_reward = reward_follow + reward_avoid + reward_progress + stagnation_penalty + disconnection_penalty + reward_exploration

    return total_reward

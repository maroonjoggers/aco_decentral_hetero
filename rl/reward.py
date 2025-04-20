# We can design the reward as we wish, depending on designed state vector

from environment import *
import numpy as np

def compute_reward(state_vector, lambda_value, prev_lambda):

    local_density, x, y, vx, vy, is_returning, progress, num_pheromones, *neighbors_poses = state_vector
    agent_pose = np.array([x, y])

    # Penalize disconnection hard
    disconnection_penalty = -3.0 if local_density == 0.0 else 0.0 

    # Reward 1: trail-following encouragement when sparse
    follow_gain = 1.0
    reward_follow = (1 - local_density) * lambda_value * follow_gain

    # Reward 2: avoidance encouragement when crowded
    avoid_gain = 1.2
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
    exploration_gain = 0.5
    reward_exploration = (1 - num_pheromones) * (1 - lambda_value) * exploration_gain

    # Penalize large lambda jumps
    fast_change_scale = 3.0
    if prev_lambda is not None:
        delta_lambda = lambda_value - prev_lambda
        fast_change_penalty = - fast_change_scale * (delta_lambda ** 2)
    else:
        fast_change_penalty = 0.0

    # Penalize being very close to neighbors
    proximity_penalty = 0.0
    proximity_threshold = 0.1  # Normalized units
    proximity_scale = 2.0       # Strength of penalty
    collision_threshold = 0.075 # estimate (?)
    neighbors_pos = np.array(neighbors_poses).reshape(-1, 2)

    for n_pos in neighbors_pos:
        if np.allclose(n_pos, agent_pose):
            continue  # Skip placeholder neighbor (same as ego position)
        dist = np.linalg.norm(agent_pose - n_pos)
        if dist < proximity_threshold:
            proximity_penalty += (proximity_threshold - dist)
        if dist < collision_threshold:
            proximity_penalty += 4.0

    proximity_penalty *= proximity_scale


    # Final reward (weight terms as needed)
    total_reward = reward_follow + reward_avoid + reward_progress + reward_exploration
    total_reward += stagnation_penalty + disconnection_penalty + fast_change_penalty 
    total_reward -= proximity_penalty

    return total_reward

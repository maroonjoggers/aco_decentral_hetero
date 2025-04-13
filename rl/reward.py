# We can design the reward as we wish, depending on designed state vector

from environment import *

def compute_reward(state_vector, lambda_value):

    local_density, x, y, vx, vy, is_returning, progress = state_vector

    # Penalize disconnection hard
    disconnection_penalty = -5.0 if local_density == 0.0 else 0.0 

    # Reward 1: trail-following encouragement when sparse
    reward_follow = (1 - local_density) * lambda_value

    # Reward 2: avoidance encouragement when crowded
    reward_avoid = local_density * (1 - lambda_value)

    # Reward 3: smooth motion or velocity alignment reward
    # reward_alignment = velocity_alignment * lambda_value * 0.1

    # Reward 4: finding food or getting back home, cumulative
    progress_scale = 3.0 # tunable
    reward_progress = progress * progress_scale

    # Penalize stagnation
    speed = np.sqrt(vx**2 + vy**2)  
    min_speed_threshold = 0.15
    stagnation_penalty_scale = 2.0
    stagnation_penalty = - stagnation_penalty_scale * max(0.0, (min_speed_threshold - speed))



    # Final reward (weight terms as needed)
    total_reward = reward_follow + reward_avoid + reward_progress + stagnation_penalty + disconnection_penalty

    return total_reward

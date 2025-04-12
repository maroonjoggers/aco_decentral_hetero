# We can design the reward as we wish, depending on designed state vector

from environment import *
from utils import NUM_AGENTS

def compute_reward(agent, env, lambda_value):

    # Penalize disconnection hard
    num_neighbors = len(env.get_agents_within_communication_radius(agent, agent.communication_radius))

    if num_neighbors == 0:
        return -10.0
    
    # Compute local_density based on number of neighbors
    local_density = num_neighbors / NUM_AGENTS

    # Reward 1: trail-following encouragement when sparse
    reward_follow = (1 - local_density) * lambda_value

    # Reward 2: avoidance encouragement when crowded
    reward_avoid = local_density * (1 - lambda_value)

    # Reward 3: smooth motion or velocity alignment reward
    # reward_alignment = velocity_alignment * lambda_value * 0.1

    # Reward 4: finding food or getting back home, cumulative
    reward_success = agent.found_goals_counter



    # Final reward (weight terms as needed)
    total_reward = reward_follow + reward_avoid + reward_success

    return total_reward

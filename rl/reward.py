# We can design the reward as we wish, depending on designed state vector

def compute_reward(state, critical_distance, desired_distance):
    distance_to_neighbor = state[0]
    exploration_score = state[1]
    number_of_neighbors = state[2]

    # Reward shaping
    exploration_gain = exploration_score * 1.0
    safe_connectivity = max(0, 1.0 - distance_to_neighbor / critical_distance)
    neighbor_bonus = number_of_neighbors * 0.1
    risky_distance_penalty = -abs(distance_to_neighbor - desired_distance)

    return exploration_gain + safe_connectivity + neighbor_bonus + risky_distance_penalty

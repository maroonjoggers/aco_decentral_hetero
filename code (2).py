# utils.py
import numpy as np

# --- Configuration Parameters ---
# Robotarium parameters (adjust as needed for your setup)
ROBOTARIUM_BOUNDARIES = np.array([-1.6, 1.6, -1.0, 1.0]) # [x_min, x_max, y_min, y_max]
TIMESTEP_SIZE = 0.033 # Robotarium default timestep (seconds)

# Environment parameters
HOME_LOCATION = [0.0, 0.0] # [x, y] - Example home location at the center
FOOD_LOCATIONS = [[1.0, 0.5], [-1.0, -0.5]] # Example food locations - list of [x, y]
OBSTACLE_LOCATIONS = [] # Define obstacles as needed (shapes and vertices) - Example: Rectangles, circles, polygons
HAZARD_LOCATIONS = [] # Define hazards as needed (shapes and vertices) - Example: Areas to avoid

# Agent parameters
NUM_AGENTS = 10 # Total number of agents in the simulation


# --- Heterogeneous Agent Trait Profiles ---
# Define different trait profiles for agents - easily extendable and modifiable
AGENT_TRAIT_PROFILES = {
    "Profile_Type_A": { # Example profile 1
        "num_agents": 5, # Number of agents with this profile - can be overridden in main script
        "sensing_radius": 0.3, # meters - Example values - ADJUST AS NEEDED
        "max_speed": 0.1, # m/s
        "initial_pheromone_strength": 1.0, # Initial pheromone strength for agents of this type
        "communication_radius": 0.5, # meters
        "pheromone_decay_rate": 0.01, # Decay rate per timestep
    },
    "Profile_Type_B": { # Example profile 2
        "num_agents": 5, # Number of agents with this profile - can be overridden
        "sensing_radius": 0.4,
        "max_speed": 0.12,
        "initial_pheromone_strength": 1.2,
        "communication_radius": 0.6,
        "pheromone_decay_rate": 0.008,
    },
    # Add more profiles as needed - easily extendable
}



# --- Helper Functions (if any) ---
# Example: Function to calculate distance between two points (already used in classes, but can be placed here for utility)
def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two 2D points.

    Args:
        point1 (numpy.ndarray): First point [x, y].
        point2 (numpy.ndarray): Second point [x, y].

    Returns:
        float: Euclidean distance.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


# --- Functions to Generate Graph Laplacians (if needed for communication or control) ---
# You can include functions like cycle_GL, completeGL, etc., from your Robotarium utilities here if you plan to use graph-based methods.
# For now, assuming communication is radius-based and graph Laplacians are not immediately needed.
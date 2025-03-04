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

# Experiment Ends when it hts this maxout time (seconds)
MAX_TIME = 120

# INITAL CONDITIONS
INTER_AGENT_DIST = 0.30

PH_LAYING_RATE = 1.0


# --- Heterogeneous Agent Trait Profiles ---
# Define different trait profiles for agents - easily extendable and modifiable
AGENT_TRAIT_PROFILES = {
    "Profile_Type_A": { # Example profile 1
        "num_agents": 5, # Number of agents with this profile - can be overridden in main script
        "sensing_radius": 0.25, # meters - Example values - ADJUST AS NEEDED
        "max_speed": 0.12, # m/s
        "initial_pheromone_strength": 1.0, # Initial pheromone strength for agents of this type
        "communication_radius": 0.5, # meters
        "pheromone_lifetime": 20.0, # Decay rate per timestep
    },
    "Profile_Type_B": { # Example profile 2
        "num_agents": 0, # Number of agents with this profile - can be overridden
        "sensing_radius": 0.25,
        "max_speed": 0.12,
        "initial_pheromone_strength": 1.2,
        "communication_radius": 0.5,
        "pheromone_lifetime": 20.0,
    },
    # Add more profiles as needed - easily extendable
}

NUM_AGENTS = sum(profile["num_agents"] for profile in AGENT_TRAIT_PROFILES.values())


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

# Determine the arrangement of Inital bots
def determineInitalConditions():
    '''
    Bots should be placed with the following attributes
        Shape --> Hexagonal Lattice
        Centered about home location
        Set distance between agents
        Random Orientations

    Returns:
        3xN numpy array which represents the IC's of each bot
    '''

    positions = []
    layers = 0

    while len(positions) < NUM_AGENTS:
        if layers == 0:
            positions.append((HOME_LOCATION[0], HOME_LOCATION[1]))  # Center bot
        else:
            for i in range(6):  # 6 directions around the hexagon
                for j in range(layers):
                    angle = np.pi / 3 * i  # 60-degree increments
                    x_offset = (layers - j) * INTER_AGENT_DIST * np.cos(angle) + j * INTER_AGENT_DIST * np.cos(angle + np.pi / 3)
                    y_offset = (layers - j) * INTER_AGENT_DIST * np.sin(angle) + j * INTER_AGENT_DIST * np.sin(angle + np.pi / 3)
                    positions.append((HOME_LOCATION[0] + x_offset, HOME_LOCATION[1] + y_offset))
                    if len(positions) >= NUM_AGENTS:
                        break
                if len(positions) >= NUM_AGENTS:
                    break
        layers += 1

    # Convert to numpy array
    positions = np.array(positions[:NUM_AGENTS]).T  # Shape (2, N)

    # Generate random orientations (-pi to pi)
    orientations = np.random.uniform(-np.pi, np.pi, NUM_AGENTS).reshape(1, NUM_AGENTS)  # Shape (1, N)

    # Stack positions and orientations
    return np.vstack((positions, orientations))  # Shape (3, N)
    


# --- Functions to Generate Graph Laplacians (if needed for communication or control) ---
# You can include functions like cycle_GL, completeGL, etc., from your Robotarium utilities here if you plan to use graph-based methods.
# For now, assuming communication is radius-based and graph Laplacians are not immediately needed.
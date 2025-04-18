# utils.py
import numpy as np

# --- Configuration Parameters ---
# Robotarium parameters (adjust as needed for your setup)
ROBOTARIUM_BOUNDARIES = np.array([-1.6, 1.6, -1.0, 1.0]) # [x_min, x_max, y_min, y_max]
TIMESTEP_SIZE = 0.033 # Robotarium default timestep (seconds)

# Environment parameters
HOME_LOCATION = [0.0, 0.0] # [x, y] - Example home location at the center
FOOD_LOCATIONS = [[1.0, 0.5], [-1.0, -0.5], [-1.5,-0.8], [1.5,-0.8], [-1.5,0.8], [1.5,0.8]] # Example food locations - list of [x, y]
OBSTACLE_LOCATIONS = [
    # {"shape": "rectangle", "center": [0.70, 0.5], "width": 0.2, "height": 0.3},
    # {"shape": "rectangle", "center": [-0.75, 0.5], "width": 0.25, "height": 0.15},
    # Add more obstacles here as dictionaries with "shape", "center", "width", "height"
]
HAZARD_LOCATIONS = [] # Define hazards as needed (shapes and vertices) - Example: Areas to avoid

# Agent parameters
AGENT_RADIUS = 0.05             #Used in obstacle detection

# Experiment Ends when it hts this maxout time (seconds)
MAX_TIME = 600

# INITAL CONDITIONS
INTER_AGENT_DIST = 0.20         # Was 0.25 during the midpoint

PH_LAYING_RATE = 1.0            # Was 0.8 during the midpoint (This is seconds between laying pheromones)

#TWO LANE PATH FORMATION
USE_PHEROMONE_LAYING_OFFSET = True
PHEROMONE_LAYING_OFFSET = 0.05

AVOID_PHEROMONE_LAYING_OFFSET = 0.125

#PULL FACTOR
PHEROMONE_PULL_FACTOR = 0.35             #Between 0 and 1, indicates strength that pheromones "pull" the bot into their location (in addition to following direction)

RANDOM_REDIRECTION_RATE = 5.0
#RANDOM_REDIRECTION_LIMITS = [np.pi/4, 2*np.pi/3]
RANDOM_REDIRECTION_LIMITS = [0.0, 2*np.pi/3]
HEADING_STD = np.pi/6

PLOTTING = False

WITH_LAMBDA = True
PLOT_LAMBDA = True
TRAINING_INTERVAL = 45 # steps, so every x*0.033 sec
TRAINING = True
USE_CHECKPOINT = False

# --- Heterogeneous Agent Trait Profiles ---
# Define different trait profiles for agents - easily extendable and modifiable
AGENT_TRAIT_PROFILES = {
    "Profile_Type_A": { # Example profile 1
        "num_agents": 5, # Number of agents with this profile - can be overridden in main script
        "sensing_radius": 0.2, # meters - Was 0.2 AT MIDPOINT
        "max_speed": 0.18, # m/s
        "initial_pheromone_strength": 1.0, # Initial pheromone strength for agents of this type
        "communication_radius": 0.8, # meters           #Was 0.2 AT MIDPOINT
        "pheromone_lifetime": 150.0, # Decay rate per timestep
    },
    "Profile_Type_B": { # Example profile 2
        "num_agents": 3, # Number of agents with this profile - can be overridden
        "sensing_radius": 0.28,
        "max_speed": 0.12,
        "initial_pheromone_strength": 0.85,
        "communication_radius": 0.4,
        "pheromone_lifetime": 80.0,
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

    RANDOM_PERTUBATION_MAX = 0.005
    RANDOM_PERTUBATION = False


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
                    if RANDOM_PERTUBATION:
                        x_offset += np.random.uniform(-RANDOM_PERTUBATION_MAX, RANDOM_PERTUBATION_MAX)
                        y_offset += np.random.uniform(-RANDOM_PERTUBATION_MAX, RANDOM_PERTUBATION_MAX)
                    #print("X OFFSET: " + str(x_offset))
                    positions.append((HOME_LOCATION[0] + x_offset, HOME_LOCATION[1] + y_offset))
                    if len(positions) >= NUM_AGENTS:
                        break
                if len(positions) >= NUM_AGENTS:
                    break
        layers += 1

    # Convert to numpy array
    positions = np.array(positions[:NUM_AGENTS]).T  # Shape (2, N)

    # Generate random orientations (-pi to pi)
    orientations = np.arctan2(positions[1], positions[0]).reshape(1, NUM_AGENTS)  # Shape (1, N)

    # Stack positions and orientations
    return np.vstack((positions, orientations))  # Shape (3, N)
    
def angle_wrapping(angle):
    """
        Ensures angle is between -pi and pi.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def communication_radius_list():
    radii = []
    for profile in AGENT_TRAIT_PROFILES.values():
        num = profile.get("num_agents", 0)
        radius = profile.get("communication_radius", 0)
        radii.extend([radius] * num)

    return radii


# DO NOT REMOVE
def is_bool_true(myBoolean):
    if myBoolean is True:
        return True
    else:
        return False






# --- Functions to Generate Graph Laplacians (if needed for communication or control) ---
# You can include functions like cycle_GL, completeGL, etc., from your Robotarium utilities here if you plan to use graph-based methods.
# For now, assuming communication is radius-based and graph Laplacians are not immediately needed.
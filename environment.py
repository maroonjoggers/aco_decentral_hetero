# environment.py
import numpy as np
from scipy.spatial import cKDTree
from agent import Agent
from utils import *

class Environment:
    def __init__(self, boundary_points, home_location, food_locations, obstacle_locations, hazard_locations, num_agents, agent_traits_profiles, agent_ICs, robotarium):
        """
        Initialize the Environment.

        Args:
            boundary_points (numpy.ndarray): Boundaries of the arena [x_min, x_max, y_min, y_max].
            home_location (numpy.ndarray): Location of the home base [x, y].
            food_locations (list): List of food locations, each as [x, y].
            obstacle_locations (list): List of obstacle locations, each defined by shape and vertices.
            hazard_locations (list): List of hazard locations, each defined by shape and vertices.
            num_agents (int): Number of agents in the environment.
            agent_traits_profiles (dict): Dictionary defining heterogeneous agent trait profiles.
        """
        self.boundaries = boundary_points # [x_min, x_max, y_min, y_max]
        self.home_location = np.array(home_location) # [x, y]
        self.food_locations = [np.array(loc) for loc in food_locations] # List of [x, y]
        self.obstacle_locations = obstacle_locations # List of obstacle definitions
        self.hazard_locations = hazard_locations # List of hazard definitions
        self.pheromone_dict = {}  # id -> pheromone object
        self.pheromone_tree = None  # KD-tree for spatial queries
        self.needs_tree_update = False
        self.agents = [] # List to store Agent objects
        self.agent_traits_profiles = agent_traits_profiles # Store trait profiles
        self.tasks_completed = 0
        self.robotarium = robotarium
        self.agent_tree = None
        self.needs_agent_tree_update = True  # Initialize as True to build tree after agents are created

        self.initialize_agents(num_agents, agent_traits_profiles, agent_ICs)
        self.update_agent_tree()  # Build initial agent tree

    def update_spatial_index(self):
        """Update the KD-tree for pheromone spatial queries"""
        if not self.needs_tree_update:
            return
        
        locations = np.array([p.location for p in self.pheromone_dict.values()])
        if len(locations) > 0:
            self.pheromone_tree = cKDTree(locations)
        else:
            self.pheromone_tree = None
        self.needs_tree_update = False

    def update_agent_tree(self):
        """Update the KD-tree for agent spatial queries"""
        if not self.needs_agent_tree_update:
            return
        
        if len(self.agents) > 0:
            positions = np.array([agent.pose[:2] for agent in self.agents])
            self.agent_tree = cKDTree(positions)
        else:
            self.agent_tree = None
        self.needs_agent_tree_update = False

    def initialize_agents(self, num_agents, agent_traits_profiles, agent_ICs):
        """
        Initialize agents with heterogeneous traits and random initial poses within boundaries.

        Args:
            num_agents (int): Number of agents to create.
            agent_traits_profiles (dict): Dictionary defining heterogeneous agent trait profiles.
        """
        profiles = list(agent_traits_profiles.keys()) # Get profile names
        num_profiles = len(profiles)
        

        agent_count = 0
        for i in range(num_profiles):
            profile_name = profiles[i]
            traits = agent_traits_profiles[profile_name] # Get traits for this profile
            agents_per_profile = traits['num_agents']

            for _ in range(agents_per_profile): # Create agents for this profile
                initial_pose = agent_ICs[:,agent_count]         #NEW IC HANDLING

                agent = Agent(agent_id=agent_count, initial_pose=initial_pose, traits=traits)
                self.agents.append(agent)
                agent_count += 1


    def update_poses(self, agent_pos_array):
        '''
        Cycle through the agents based on their ID number and tell them what their new pose is
        This info comes directly from the robotarium
        '''

        for idx, agent in enumerate(self.agents):
            agent.pose = agent_pos_array[:, idx]

    def update_velocities(self, agent_vel_array):
        '''
        Cycle through the agents based on their ID number and tell them what their new pose is
        This info comes directly from the robotarium
        '''

        for idx, agent in enumerate(self.agents):
            agent.velocity_vector = agent_vel_array[:, idx]



    def get_random_pose_in_bounds(self):
        """
        Generate a random pose (x, y, theta) within the environment boundaries.

        Returns:
            numpy.ndarray: Random pose [x, y, theta].
        """
        #TODO: A better replacement for this would be to distribute them evenely but closely at or very near the home location. Envision 2 (somewhat conflicting) goals for this:
        # 1. Get all agents IC's to start as close to the home location as possible
        # 2. No 2 agents' IC's should be too close together

        x_min, x_max, y_min, y_max = self.boundaries
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        theta = np.random.uniform(0, 2 * np.pi)  # Random orientation
        return np.array([x, y, theta])


    def create_pheromone(self, agent_id, type, location, direction, strength, lifeTime):
        """
        Create a new pheromone object.

        Args:
            agent_id (int): ID of the agent that laid the pheromone.
            type (str): Type of pheromone ("Return Home", "To Food", "Avoidance").
            location (numpy.ndarray): Location [x, y] of the pheromone.
            direction (float): Direction of the pheromone (orientation of agent laying it).
            strength (float): Initial strength of the pheromone.
            decay_rate (float): Rate at which the pheromone decays.

        Returns:
            Pheromone: A new Pheromone object.
        """
        return Pheromone(agent_id, type, location, direction, strength, lifeTime)


    def add_pheromone(self, pheromone):
        """
        Add a pheromone to the environment's pheromone dictionary and mark tree for update.

        Args:
            pheromone (Pheromone): The Pheromone object to add.
        """
        self.pheromone_dict[pheromone.id] = pheromone
        self.needs_tree_update = True

    def decay_pheromones(self):
        """
        Decay all pheromones in the environment and remove those with strength <= 0.
        Uses batch processing for efficiency.
        """
        dead_pheromones = []
        for p_id, pheromone in self.pheromone_dict.items():
            pheromone.decay()
            if pheromone.strength <= 0:
                dead_pheromones.append(p_id)
        
        # Batch remove dead pheromones
        for p_id in dead_pheromones:
            del self.pheromone_dict[p_id]
        
        if dead_pheromones:
            self.needs_tree_update = True

    def get_nearby_pheromones(self, agent_location, sensing_radius, agent_pheromone_map):
        """
        Get a list of pheromones from the environment that are within the agent's sensing radius
        AND are not already in the agent's pheromone map.

        Args:
            agent_location (numpy.ndarray): Agent's location [x, y].
            sensing_radius (float): Agent's sensing radius.
            agent_pheromone_map (dict): The agent's current pheromone map.

        Returns:
            list: List of Pheromone objects from the environment that are nearby and new to the agent.
        """
        self.update_spatial_index()
        if self.pheromone_tree is None:
            return []
            
        nearby_indices = self.pheromone_tree.query_ball_point(
            agent_location, sensing_radius
        )
        
        # Filter known pheromones
        known_ids = set(agent_pheromone_map.keys())
        return [
            self.pheromone_dict[idx] 
            for idx in nearby_indices 
            if idx not in known_ids
        ]

    def get_agents_within_communication_radius(self, agent, communication_radius):
        """
        Get a list of neighboring agents within the communication radius of a given agent.

        Args:
            agent (Agent): The agent to find neighbors for.
            communication_radius (float): The communication radius.

        Returns:
            list: List of neighboring Agent objects.
        """
        self.update_agent_tree()
        if self.agent_tree is None:
            return []
            
        agent_pos = agent.pose[:2]
        nearby_indices = self.agent_tree.query_ball_point(
            agent_pos, communication_radius
        )
        
        return [
            self.agents[idx] 
            for idx in nearby_indices 
            if self.agents[idx].id != agent.id
        ]


    def get_nearby_food(self, agent_location, sensing_radius):
        """
        Check if food sources are within the agent's sensing radius.

        Args:
            agent_location (numpy.ndarray): Agent's location [x, y].
            sensing_radius (float): Agent's sensing radius.

        Returns:
            bool: True if food is nearby, False otherwise.
        """
        for food_location in self.food_locations:
            distance = np.linalg.norm(food_location - agent_location) # Distance calculation
            if distance <= sensing_radius:
                return True # Food found nearby
        return False # No food nearby


    def is_nearby_home(self, agent_location, sensing_radius):
        """
        Check if the home location is within the agent's sensing radius.

        Args:
            agent_location (numpy.ndarray): Agent's location [x, y].
            sensing_radius (float): Agent's sensing radius.

        Returns:
            bool: True if home is nearby, False otherwise.
        """
        distance = np.linalg.norm(self.home_location - agent_location) # Distance to home
        return distance <= sensing_radius*1.5


    def get_nearby_obstacles(self, agent_location, sensing_radius):
        """
        Check for obstacles within the agent's sensing radius.
        For now, checks if the agent's center is inside a rectangular obstacle.
        More sophisticated checks (e.g., considering agent radius) can be added.
        """
        for obstacle in self.obstacle_locations:
            if obstacle["shape"] == "rectangle":
                center_x, center_y = obstacle["center"]
                width = obstacle["width"]
                height = obstacle["height"]
                x_min_obs = center_x - width / 2
                x_max_obs = center_x + width / 2
                y_min_obs = center_y - height / 2
                y_max_obs = center_y + height / 2

                # Calculate the shortest distance between the agent's center and the obstacle's bounding box
                distance_x = max(x_min_obs - agent_location[0], 0, agent_location[0] - x_max_obs)
                distance_y = max(y_min_obs - agent_location[1], 0, agent_location[1] - y_max_obs)
                distance = np.sqrt(distance_x**2 + distance_y**2)

                # Check if the distance is less than agent_radius
                if distance <= sensing_radius:
                    obstacle_angle = np.arctan2(-distance_y, -distance_x)
                    return True, obstacle_angle
        return False, None


    def get_nearby_hazards(self, agent_location, sensing_radius):
        """
        Check for hazards within the agent's sensing radius.

        Args:
            agent_location (numpy.ndarray): Agent's location [x, y].
            sensing_radius (float): Agent's sensing radius.

        Returns:
            bool: True if hazards are nearby, False otherwise.
        """
        # --- Hazard detection logic using hazard_locations (shapes and vertices) ---
        # TODO: IMPLEMENTATION NEEDED - Placeholder for now
        return False # Placeholder - Replace with actual hazard detection


    #UNUSED - DO NOT USE
    def update_agent_poses(self, agent_velocities):
        """
        Update agents' poses based on provided velocities (Single Integrator) and environment boundaries.

        Args:
            agent_velocities (numpy.ndarray): Array of agent velocities in SI form [vx, vy] for each agent.
        """
        for i in range(len(self.agents)):
            agent = self.agents[i]
            velocity = agent_velocities[:, i] # Get velocity for this agent
            current_pose = agent.get_pose() # Get current pose

            # --- Simple Euler integration for pose update ---
            new_pose = current_pose.copy()
            new_pose[:2] += velocity * 0.033 # Assuming 0.033s timestep (Robotarium default) - make timestep variable?

            # --- Boundary Constraint Enforcement (simple reflection) ---
            x_min, x_max, y_min, y_max = self.boundaries
            if new_pose[0] < x_min:
                new_pose[0] = x_min # Clamp within boundaries
                velocity[0] = 0 # Stop x-velocity
            elif new_pose[0] > x_max:
                new_pose[0] = x_max
                velocity[0] = 0
            if new_pose[1] < y_min:
                new_pose[1] = y_min
                velocity[1] = 0
            elif new_pose[1] > y_max:
                new_pose[1] = y_max
                velocity[1] = 0

            print(f"Agent {agent.id} moved from {current_pose} to {new_pose}")
            agent.set_pose(new_pose) # Update agent's pose


    def get_agent_poses(self):
        """
        Get poses of all agents in the environment.

        Returns:
            numpy.ndarray: Array of agent poses, each as [x, y, theta]. Shape (3, N_agents).
        """
        poses = np.array([agent.get_pose() for agent in self.agents]).T # Transpose to get 3xN array
        return poses


    def get_all_pheromones(self):
        """
        Get a list of all pheromone objects currently in the environment.

        Returns:
            list: List of Pheromone objects.
        """
        return self.pheromone_dict.values()


    def get_agents(self):
        """S
        Get a list of all Agent objects in the environment.

        Returns:
            list: List of Agent objects.
        """
        return self.agents
    
    def get_num_neighbors(self, agent):
        """
        Compute the number of neighboring agents within the communication radius.

        Returns:
            int: Number of neighbors
        """
        neighbors = self.get_agents_within_communication_radius(agent, agent.communication_radius)
        return len(neighbors)
    
        
    def get_state_vector(self, agent):
        """
        Decentralized state vector for the agent (7-dim) (normalized)
        - Local agent density (neighbors within communication radius)
        - Agent's own pose (x, y)
        - Agent's velocity (vx, vy)
        - Foraging/returning state (binary)
        - Found goals progress
        """
        state = []

        # 1. Local agent density (number of neighbors within communication radius)
        local_density = self.get_num_neighbors(agent)
        max_neighbors = NUM_AGENTS - 1 # for normalization
        state.append(local_density / max_neighbors if max_neighbors > 0 else 0.0)

         # 2. Agent's own pose (x, y)
        x_min, x_max, y_min, y_max = ROBOTARIUM_BOUNDARIES
        norm_x = 2 * (agent.pose[0] - x_min) / (x_max - x_min) - 1
        norm_y = 2 * (agent.pose[1] - y_min) / (y_max - y_min) - 1
        state.extend([norm_x, norm_y])

        # 3. Agent's velocity (vx, vy)
        norm_vx = agent.velocity_vector[0] / agent.max_speed
        norm_vy = agent.velocity_vector[1] / agent.max_speed
        state.extend([norm_vx, norm_vy])

        # 4. Foraging / returning state (binary)
        is_returning = 1 if agent.state == "Returning" else 0
        state.append(is_returning)

        # 5. Found goals progress
        progress_sigmoid = 1 / (1 + np.exp(-agent.found_goals_counter))
        progress_norm = (progress_sigmoid - 0.5) * 2 # rescaled to [0,1]
        progress_norm = np.clip(progress_norm, 0.0, 1.0)
        state.append(progress_norm)

        return np.array(state, dtype=np.float32)


# --- Pheromone Class (Inner class within Environment or separate file pheromone.py - CHOOSE ONE) ---
class Pheromone:
    # Add a class variable to track the next available ID
    _next_id = 0

    def __init__(self, agent_id, type, location, direction, strength, lifeTime):
        """
        Initialize a Pheromone object.

        Args:

            agent_id (int): ID of the agent that laid the pheromone.
            type (str): Type of pheromone ("Return Home", "To Food", "Avoidance").
            location (numpy.ndarray): Location [x, y] of the pheromone.
            direction (float): Direction of the pheromone.
            strength (float): Current strength of the pheromone.
            decay_rate (float): Rate at which the pheromone decays per timestep.
        """
        self.id = Pheromone._next_id  # Use simple integer ID
        Pheromone._next_id += 1
        self.agent_id = agent_id
        self.type = type
        self.location = np.array(location) # [x, y]
        self.direction = direction # (global) Theta relative to the positive x-axis
        self.strength = strength # Initial and current strength
        self.decay_rate = strength/lifeTime # Decay rate per timestep

    def decay(self):
        """
        Decrease the pheromone strength based on its decay rate.
        """
        self.strength -= self.decay_rate * PH_LAYING_RATE # Linear decay - change decay function if needed
        self.strength = max(0, self.strength) # Ensure strength doesn't go below zeroSSS
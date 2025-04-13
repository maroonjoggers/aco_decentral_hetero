# environment.py
import numpy as np
from agent import Agent
import uuid # For generating unique IDs for pheromones
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
        self.food_locations = [np.array(loc) for loc in food_locations] # List of [x, y]    #TODO: Why is this different than obstacles and hazards? If its passed as a list then why are we parsing the list just to build a new one?
        self.obstacle_locations = obstacle_locations # List of obstacle definitions
        self.hazard_locations = hazard_locations # List of hazard definitions
        self.pheromones = [] # List to store pheromone objects in the environment
        self.agents = [] # List to store Agent objects
        self.agent_traits_profiles = agent_traits_profiles # Store trait profiles
        self.tasks_completed = 0
        self.robotarium = robotarium

        self.initialize_agents(num_agents, agent_traits_profiles, agent_ICs)


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


    def updatePoses(self, agent_Pos_array):
        '''
        Cycle through the agents based on their ID number and tell them what their new pose is
        This info comes directly from the robotarium
        '''

        for idx, agent in enumerate(self.agents):
            agent.pose = agent_Pos_array[:, idx]



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
        pheromone_id = uuid.uuid4() # Generate unique ID
        return Pheromone(pheromone_id, agent_id, type, location, direction, strength, lifeTime)


    def add_pheromone(self, pheromone):
        """
        Add a pheromone to the environment's pheromone list.

        Args:
            pheromone (Pheromone): The Pheromone object to add.
        """
        self.pheromones.append(pheromone)


    def decay_pheromones(self):
        """
        Decay all pheromones in the environment and remove those with strength <= 0.
        """
        updated_pheromones = []
        for p in self.pheromones:
            p.decay()
            if p.strength > 0: # Keep pheromones with positive strength
                updated_pheromones.append(p)
        self.pheromones = updated_pheromones


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
        return distance <= sensing_radius


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
                if distance <= AGENT_RADIUS:
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


    def get_nearby_pheromones(self, agent_location, sensing_radius, agent_pheromone_map):
        """
        Get a list of pheromones from the environment that are within the agent's sensing radius AND are not already in the agent's pheromone map.
        This prevents an agent from adding pheromones to its map multiple times from the environment.

        Args:
            agent_location (numpy.ndarray): Agent's location [x, y].
            sensing_radius (float): Agent's sensing radius.
            agent_pheromone_map (list): The agent's current pheromone map (list of Pheromone objects).

        Returns:
            list: List of Pheromone objects from the environment that are nearby and new to the agent.
        """
        nearby_pheromones = []
        for pheromone in self.pheromones:
            distance = np.linalg.norm(pheromone.location - agent_location) # Distance calculation
            if distance <= sensing_radius:
                is_known_pheromone = False
                for p_local in agent_pheromone_map:
                    if pheromone.id == p_local.id: # Check if already in agent's map by ID
                        is_known_pheromone = True
                        break
                if not is_known_pheromone:
                    nearby_pheromones.append(pheromone) # Add only new pheromones

        # print(f"Agent at {agent_location} detected {len(nearby_pheromones)} pheromones")

        return nearby_pheromones


    def get_agents_within_communication_radius(self, agent, communication_radius):
        """
        Get a list of neighboring agents within the communication radius of a given agent.

        Args:
            agent (Agent): The agent to find neighbors for.
            communication_radius (float): The communication radius.

        Returns:
            list: List of neighboring Agent objects.
        """
        neighbors = []
        for other_agent in self.agents:
            if other_agent.id != agent.id: # Exclude self
                distance = np.linalg.norm(agent.pose[:2] - other_agent.pose[:2]) # Distance between agents (xy only)
                if distance <= communication_radius:
                    neighbors.append(other_agent)
        return neighbors


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
        return self.pheromones


    def get_agents(self):
        """S
        Get a list of all Agent objects in the environment.

        Returns:
            list: List of Agent objects.
        """
        return self.agents


# --- Pheromone Class (Inner class within Environment or separate file pheromone.py - CHOOSE ONE) ---
class Pheromone:
    def __init__(self, pheromone_id, agent_id, type, location, direction, strength, lifeTime):
        """
        Initialize a Pheromone object.

        Args:
            pheromone_id (uuid.UUID): Unique ID for the pheromone.
            agent_id (int): ID of the agent that laid the pheromone.
            type (str): Type of pheromone ("Return Home", "To Food", "Avoidance").
            location (numpy.ndarray): Location [x, y] of the pheromone.
            direction (float): Direction of the pheromone.
            strength (float): Current strength of the pheromone.
            decay_rate (float): Rate at which the pheromone decays per timestep.
        """
        self.id = pheromone_id # Unique ID for pheromone identification
        self.agent_id = agent_id # ID of agent that created it
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
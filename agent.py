# agent.py
import numpy as np
import environment
# from environment import Environment
import copy
from utils import *
import random

class Agent:
    def __init__(self, agent_id, initial_pose, traits):
        """
        Initialize an Agent.

        Args:
            agent_id (int): Unique identifier for the agent.
            initial_pose (numpy.ndarray): Initial pose (x, y, theta).
            traits (dict): Dictionary of heterogeneous traits (sensing_radius, etc.).
        """
        self.id = agent_id
        self.pose = initial_pose  # [x, y, theta]  #TODO: We don't need to feed it its OG position. We tell the robotarium where to start them off and their position should ALWAYS be read, not directly written
        self.state = "Foraging"  # Initial state
        self.pheromone_map = [] # List of pheromone objects perceived by this agent
        self.age = 0

        # Heterogeneous Traits - loaded from traits dictionary
        self.sensing_radius = traits.get('sensing_radius')
        self.max_speed = traits.get('max_speed')
        self.initial_pheromone_strength = traits.get('initial_pheromone_strength')
        self.communication_radius = traits.get('communication_radius')
        self.pheromone_lifetime = traits.get('pheromone_lifetime') # New trait
        self.random_direction_change_timer = 0

        # self.motion_update_interval = np.random.randint(5, 20)  # Update motion every 5-20 steps
        # self.motion_update_counter = 0  # Counter to track updates
        # self.current_velocity = np.array([0.0, 0.0])  # Stores last computed velocity


    def check_environment(self, environment):
        """
        Check for food, home, obstacles, and hazards in the environment within the agent's sensing radius.
        Update agent's state based on environmental conditions.

        Args:
            environment (Environment): The environment object.
        """
        nearby_food = environment.get_nearby_food(self.pose[:2], self.sensing_radius) # Returns true if we see food
        nearby_home = environment.is_nearby_home(self.pose[:2], self.sensing_radius) # Returns true if we see the home
        nearby_obstacles = environment.get_nearby_obstacles(self.pose[:2], self.sensing_radius) # TODO: NOT IMPLEMENTED. For now returns False
        nearby_hazards = environment.get_nearby_hazards(self.pose[:2], self.sensing_radius) # TODO: NOT IMPLEMENTED. For now returns False

        #TODO: What is the purpose of this here? It's not used and the function itself doesn't even make much sense. Probably get rid of it
        nearby_pheromones = environment.get_nearby_pheromones(self.pose[:2], self.sensing_radius, self.pheromone_map) # TODO: CHECK THIS

        # --- State Update Logic based on environment checks ---
        # TODO: Right now this translates to: If we sense our goal then we immeditaly attain it. Might be okay for now but may want to reconsider
        if self.state == "Foraging":
            if nearby_food:
                self.state = "Returning"
        elif self.state == "Returning":
            if nearby_home:
                self.state = "Foraging"
                environment.tasks_completed +=1
                print(f"Tasks completed: {environment.tasks_completed}")

        # --- Handle obstacle and hazard avoidance based on nearby_obstacles and nearby_hazards ---
        # This might involve setting avoidance pheromones or directly influencing velocity
        # # TODO: This won't run right now since both will be false
        if nearby_obstacles or nearby_hazards:
            self.update_pheromone_map_own(environment=environment, avoidance=True) # Example: Lay avoidance pheromone


    def update_pheromone_map_own(self, environment, avoidance=False):
        """
        Lay down pheromones based on the agent's current state and decay existing pheromones in its map.

        Args:
            environment (Environment): The environment object to lay pheromones in.
            state (str): Current agent state (determines which pheromone to lay).
        """
        # --- Pheromone Laying Logic ---
        pheromone_type = None


        if self.state == "Foraging":
            pheromone_type = "Return Home"
        elif self.state == "Returning":
            pheromone_type = "To Food"
        
        if avoidance:
            pheromone_type = "Avoidance"

        if pheromone_type:
            pheromone = environment.create_pheromone(
                agent_id=self.id,
                type=pheromone_type,
                location=self.pose[:2].copy(),
                direction=angle_wrapping(np.pi+self.pose[2]),        #Reverse the direction  
                strength=self.initial_pheromone_strength,
                lifeTime=self.pheromone_lifetime
            )

            # Add pheromone to agent's local pheromone map
            self.pheromone_map.append(pheromone)  
            print(f"Agent {self.id} added {pheromone_type} pheromone to local map at {self.pose[:2]}")

            # Optionally, still add to the environment
            environment.add_pheromone(pheromone)  

        # --- Pheromone Decay for the agent's local pheromone map ---
        updated_pheromone_map = []
        for p in self.pheromone_map:
            p.decay()  
            if p.strength > 0:
                updated_pheromone_map.append(p)
            else:                                           #If the pheromone died, remove it from the global list
                for ph in environment.pheromones:
                    if ph.id == p.id:
                        environment.pheromones.remove(ph)

        self.pheromone_map = updated_pheromone_map



    def update_pheromone_map_shared(self, environment, neighbors):
        """
        Update the agent's pheromone map by incorporating information from neighboring agents.

        Args:
            environment (Environment): The environment object.
            neighbors (list): List of neighboring Agent objects within communication radius.
        """
        for neighbor in neighbors:
            shared_pheromones = neighbor.pheromone_map     #TODO: This is wrong, we want ALL of the pheromones known by the other agent to be shared. Rest here looks decent I think
            for p_shared in shared_pheromones:
                is_new_pheromone = True
                for p_local in self.pheromone_map:
                    if p_shared.id == p_local.id: # Check if already in local map by ID
                        is_new_pheromone = False
                        break
                if is_new_pheromone:
                    self.pheromone_map.append(copy.copy(p_shared)) # Add new pheromone to local map



    def get_perceived_pheromones(self, center_location, radius):
        """
        Returns a list of pheromone objects from the agent's map that are within a given radius of a location.
        Used for sharing pheromone information with neighbors.

        Args:
            center_location (numpy.ndarray): The center location (x, y) to check around.
            radius (float): The radius to check within.

        Returns:
            list: List of Pheromone objects within the radius.
        """
        perceived_pheromones = []
        for pheromone in self.pheromone_map:
            distance = np.linalg.norm(pheromone.location - center_location) # Distance calculation
            if distance <= radius:
                perceived_pheromones.append(pheromone)
        return perceived_pheromones


    def determine_velocity_inputs_aco(self, environment, current_time):
        """
        Determine velocity inputs (Single Integrator form) based on ACO logic and pheromone map.

        Args:
            environment (Environment): The environment object (may be needed to get global info).

        Returns:
            numpy.ndarray: Velocity inputs in Single Integrator form [vx, vy].
        """
        #TODO: WHAT TO DO IF NO PHEROMONES --> Random Movement

        scaling = 0.75

        # --- ACO Velocity Calculation Logic ---
        # 1. Get relevant pheromones from agent's map and environment
        relevant_pheromones = self.get_relevant_pheromones_for_state(environment)       #TODO: Needs work, check function

        # 2. Calculate resultant pheromone vector (sum of vector contributions of pheromones)
        resultant_vector = np.array([0.0, 0.0])
        for pheromone in relevant_pheromones:
            pheromone_vector = self.calculate_pheromone_vector(pheromone) # Method to define pheromone vector contribution  TODO: Needs work, check function
            resultant_vector += pheromone_vector

        # 3. Normalize resultant vector if needed (to limit speed or direction influence)
        # TODO: I think we should just explicitly always do this
        if np.linalg.norm(resultant_vector) > 0:
            resultant_vector = resultant_vector / np.linalg.norm(resultant_vector)
            movement_direction = np.arctan2(resultant_vector[1], resultant_vector[0])
            speed = np.linalg.norm(resultant_vector)
        else:
            movement_direction = self.pose[2]
            speed = self.max_speed * scaling

        # 4. Add probabilistic element for exploration (e.g., random deviation) - IMPLEMENTATION NEEDED
        # if np.linalg.norm(resultant_vector) < 1e-3:
        #     random_angle = np.random.uniform(0, 2 * np.pi)  # Pick a random direction
        #     resultant_vector = np.array([np.cos(random_angle), np.sin(random_angle)]) *2  # Small step
        #     print(f"Agent {self.id} using random movement: {resultant_vector}")
        if current_time - self.random_direction_change_timer >= RANDOM_REDIRECTION_RATE:
            movement_direction += random.uniform(*RANDOM_REDIRECTION_LIMITS)
            movement_direction = angle_wrapping(movement_direction)
            self.random_direction_change_timer = current_time
        
        gauss_noise = np.random.normal(loc=movement_direction, scale=HEADING_STD )
        resultant_vector = np.array([np.cos(gauss_noise), np.sin(gauss_noise)]) * speed

        velocity_input = resultant_vector
        # 4. Add probabilistic element for exploration (e.g., random deviation)
        # TODO: IMPLEMENTATION NEEDED
        velocity_input = resultant_vector # Placeholder - replace with probabilistic ACO velocity

        # 5. Limit velocity magnitude based on max_speed trait
        # TODO: This is redundant to step 3, but I think its better... maybe just remove step 3?
        speed_magnitude = np.linalg.norm(velocity_input)
        if speed_magnitude > self.max_speed:
            velocity_input = velocity_input * (self.max_speed / speed_magnitude)


        return velocity_input


    def get_relevant_pheromones_for_state(self, environment):
        """
        Determine which pheromone types are relevant for the agent's current state.

        Args:
            environment (Environment): The environment object (may be needed to access global pheromone list).

        Returns:
            list: List of relevant Pheromone objects from the agent's pheromone map.
        """
        #TODO: There is no check of the pheromones which are actually within the pheromone sensing radius, its literally going to look at all the pheromones lmao

        relevant_pheromones = []
        if self.state == "Foraging":
            # Consider "To Food" and "Avoidance" pheromones
            for p in self.pheromone_map:
                if p.type in ["To Food", "Avoidance"]: # Consider Return Home to find general direction back
                    relevant_pheromones.append(p)
        elif self.state == "Returning":
            # Consider "Return Home" and "Avoidance" pheromones
            for p in self.pheromone_map:
                if p.type in ["Return Home", "Avoidance"]:
                    relevant_pheromones.append(p)
        return relevant_pheromones


    def calculate_pheromone_vector(self, pheromone):
        """
        Calculate the vector contribution of a single pheromone to the agent's movement.

        Args:
            pheromone (Pheromone): The pheromone object.

        Returns:
            numpy.ndarray: 2D vector representing pheromone influence [vx, vy].
        """

        #Vectorized form of the pheromone
        pheromone_vector = pheromone.strength * np.array([np.cos(pheromone.direction), np.sin(pheromone.direction)])

        #TODO FIXME
        #vectorFromAgentToPh = pheromone.location - self.pose[:2]
        #vectorFromAgentToPh /= np.linalg.norm(vectorFromAgentToPh) 

        '''
        # --- Define how each pheromone type influences movement direction and strength ---
        pheromone_vector = np.array([0.0, 0.0])
        if pheromone.type == "To Food":
            # Vector towards food source (pheromone location) - EXAMPLE IMPLEMENTATION -  ADJUST LOGIC
            direction_vector = pheromone.location - self.pose[:2]
            magnitude = pheromone.strength # Strength of pheromone influence
            if np.linalg.norm(direction_vector) > 0:
                pheromone_vector = magnitude * (direction_vector / np.linalg.norm(direction_vector)) # Normalize direction

        elif pheromone.type == "Return Home":
            # Vector towards home - EXAMPLE IMPLEMENTATION - ADJUST LOGIC
            direction_vector = environment.home_location - self.pose[:2] # Assuming home_location is accessible from Agent or passed
            magnitude = pheromone.strength
            if np.linalg.norm(direction_vector) > 0:
                pheromone_vector = magnitude * (direction_vector / np.linalg.norm(direction_vector))

        elif pheromone.type == "Avoidance":
            # Vector away from avoidance pheromone - EXAMPLE IMPLEMENTATION - ADJUST LOGIC
            direction_vector = self.pose[:2] - pheromone.location # Flee direction
            magnitude = pheromone.strength
            if np.linalg.norm(direction_vector) > 0:
                pheromone_vector = magnitude * (direction_vector / np.linalg.norm(direction_vector))
        '''


        return pheromone_vector


    def update_age(self):
        """
        Increment the agent's age.
        """
        #TODO: Probably want to do this on a time basis? Could updated by 0.033 seconds each robotarium iteration

        self.age += 0.033


    def set_pose(self, new_pose):
        """
        Set the agent's pose. Used for Robotarium updates.

        Args:
            new_pose (numpy.ndarray): The new pose [x, y, theta].
        """
        self.pose = new_pose


    def get_pose(self):
        """
        Get the agent's current pose.

        Returns:
            numpy.ndarray: The pose [x, y, theta].
        """
        return self.pose
# agent.py
import numpy as np
import environment
# from environment import Environment
import copy
from utils import *
import random

class Agent:
    def __init__(
        self, agent_id, initial_pose, traits):
        """
        Initialize an Agent.

        Args:
            agent_id (int): Unique identifier for the agent.
            initial_pose (numpy.ndarray): Initial pose (x, y, theta).
            traits (dict): Dictionary of heterogeneous traits (sensing_radius, etc.).
        """
        self.id = agent_id
        self.pose = initial_pose  # [x, y, theta]
        self.state = "Foraging"  # Initial state
        self.pheromone_map = {}  # Dictionary for O(1) lookups
        self.age = 0

        # Heterogeneous Traits - loaded from traits dictionary
        self.sensing_radius = traits.get('sensing_radius')
        self.max_speed = traits.get('max_speed')
        self.initial_pheromone_strength = traits.get('initial_pheromone_strength')
        self.communication_radius = traits.get('communication_radius')
        self.pheromone_lifetime = traits.get('pheromone_lifetime')
        self.random_direction_change_timer = 0

        self.velocity_vector = [0, 0]
        self.found_goals_counter = 0

        # Pre-allocate vectors for reuse
        self._temp_vector = np.zeros(2)
        self._direction_vector = np.zeros(2)

    def check_environment(self, environment):
        """
        Check the environment for obstacles, hazards, and food/goal.
        Updates agent state and pheromone map accordingly.
        """
        nearby_food = environment.get_nearby_food(self.pose[:2], self.sensing_radius) # Returns true if we see food
        nearby_home = environment.is_nearby_home(self.pose[:2], self.sensing_radius) # Returns true if we see the home
        nearby_obstacles, obstacle_angle = environment.get_nearby_obstacles(self.pose[:2], self.sensing_radius) # TODO: NOT IMPLEMENTED. For now returns False
        nearby_hazards = environment.get_nearby_hazards(self.pose[:2], self.sensing_radius) # TODO: NOT IMPLEMENTED. For now returns False

        #TODO: What is the purpose of this here? It's not used and the function itself doesn't even make much sense. Probably get rid of it
        #nearby_pheromones = environment.get_nearby_pheromones(self.pose[:2], self.sensing_radius, self.pheromone_map) # TODO: CHECK THIS

        # --- State Update Logic based on environment checks ---
        # TODO: Right now this translates to: If we sense our goal then we immeditaly attain it. Might be okay for now but may want to reconsider
        if self.state == "Foraging":
            if nearby_food:
                self.state = "Returning"
                self.found_goals_counter += 1
        elif self.state == "Returning":
            if nearby_home:
                self.state = "Foraging"
                self.found_goals_counter += 1
                environment.tasks_completed +=1
                print(f"Tasks completed: {environment.tasks_completed}")

        # --- Handle obstacle and hazard avoidance based on nearby_obstacles and nearby_hazards ---
        # This might involve setting avoidance pheromones or directly influencing velocity
        # # TODO: This won't run right now since both will be false
        # if nearby_obstacles or nearby_hazards:
        #     self.update_pheromone_map_own(environment=environment, avoidance=True, avoidance_angle=obstacle_angle) # Example: Lay avoidance pheromone


    def update_pheromone_map_own(self, environment, avoidance=False, avoidance_angle = None):
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
            if pheromone_type != "Avoidance":
                if USE_PHEROMONE_LAYING_OFFSET:
                    ph_x = self.pose[0] - np.sin(self.pose[2]) * PHEROMONE_LAYING_OFFSET
                    ph_y = self.pose[1] + np.cos(self.pose[2]) * PHEROMONE_LAYING_OFFSET
                    ph_location = [ph_x, ph_y]
                else:
                    ph_location = [self.pose[0], self.pose[1]]
            else:
                ph_x = self.pose[0] + np.cos(avoidance_angle) * AVOID_PHEROMONE_LAYING_OFFSET
                ph_y = self.pose[1] + np.sin(avoidance_angle) * AVOID_PHEROMONE_LAYING_OFFSET
                ph_location = [ph_x, ph_y]
            
            pheromone = environment.create_pheromone(
                agent_id=self.id,
                type=pheromone_type,
                location=ph_location,
                direction=angle_wrapping(np.pi+self.pose[2]),        #Reverse the direction  
                strength=self.initial_pheromone_strength * (2.0 if avoidance else 1.0), # Make avoidance pheromones stronger?
                lifeTime=self.pheromone_lifetime * (0.5 if avoidance else 1.0) # Make avoidance pheromones decay faster?
            )

            # Add pheromone to agent's local pheromone map
            self.pheromone_map[pheromone.id] = pheromone
            print(f"Agent {self.id} added {pheromone_type} pheromone to local map at {self.pose[:2]}")

            # Add to the environment
            environment.add_pheromone(pheromone)  

        # --- Pheromone Decay for the agent's local pheromone map ---
        updated_pheromone_map = {}
        for p_id, p in self.pheromone_map.items():
            p.decay()  
            if p.strength > 0:
                updated_pheromone_map[p_id] = p
            else:                                           #If the pheromone died, remove it from the global list
                if p_id in environment.pheromone_dict:
                    del environment.pheromone_dict[p_id]
                    environment.needs_tree_update = True

        self.pheromone_map = updated_pheromone_map



    def update_pheromone_map_shared(self, environment, neighbors):
        """
        Update the agent's pheromone map by incorporating information from neighboring agents.

        Args:
            environment (Environment): The environment object.
            neighbors (list): List of neighboring Agent objects within communication radius.
        """
        for neighbor in neighbors:
            for p_id, p_shared in neighbor.pheromone_map.items():
                if p_id not in self.pheromone_map:
                    self.pheromone_map[p_id] = copy.copy(p_shared)



    def get_perceived_pheromones(self, relevant_pheromones, center_location, radius):
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
        for pheromone in relevant_pheromones:
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
        relevant_pheromones = self.get_relevant_pheromones_for_state(environment)
        relevant_pheromones = self.get_perceived_pheromones(relevant_pheromones, self.pose[:2], self.sensing_radius)

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
            speed = self.max_speed

        # 4. Add probabilistic element for exploration (e.g., random deviation) - IMPLEMENTATION NEEDED
        # if np.linalg.norm(resultant_vector) < 1e-3:
        #     random_angle = np.random.uniform(0, 2 * np.pi)  # Pick a random direction
        #     resultant_vector = np.array([np.cos(random_angle), np.sin(random_angle)]) *2  # Small step
        #     print(f"Agent {self.id} using random movement: {resultant_vector}")
            # if current_time - self.random_direction_change_timer >= RANDOM_REDIRECTION_RATE:
            #     movement_direction += (random.uniform(*RANDOM_REDIRECTION_LIMITS) * random.choice([-1, 1]))
            #     movement_direction = angle_wrapping(movement_direction)
            #     self.random_direction_change_timer = current_time
            
        gauss_noise = np.random.normal(loc=movement_direction, scale=HEADING_STD )
        resultant_vector = np.array([np.cos(gauss_noise), np.sin(gauss_noise)]) * speed

        velocity_input = resultant_vector
        # 4. Add probabilistic element for exploration (e.g., random deviation)
        # TODO: IMPLEMENTATION NEEDED
        # velocity_input = resultant_vector # Placeholder - replace with probabilistic ACO velocity

        # 5. Limit velocity magnitude based on max_speed trait
        # TODO: This is redundant to step 3, but I think its better... maybe just remove step 3?
        speed_magnitude = np.linalg.norm(velocity_input)
        # if speed_magnitude > self.max_speed:
        velocity_input = velocity_input * (self.max_speed / speed_magnitude)
        # velocity_input /= np.linalg.norm(velocity_input)


        return velocity_input


    def get_relevant_pheromones_for_state(self, environment):
        """
        Determine which pheromone types are relevant for the agent's current state.

        Args:
            environment (Environment): The environment object (may be needed to access global pheromone list).

        Returns:
            list: List of relevant Pheromone objects from the agent's pheromone map.
        """

        relevant_pheromones = []
        if self.state == "Foraging":
            # Consider "To Food" and "Avoidance" pheromones
            for p in self.pheromone_map.values():
                if p.type in ["To Food", "Avoidance"]: # Consider Return Home to find general direction back
                    relevant_pheromones.append(p)
        elif self.state == "Returning":
            # Consider "Return Home" and "Avoidance" pheromones
            for p in self.pheromone_map.values():
                if p.type in ["Return Home", "Avoidance"]:
                    relevant_pheromones.append(p)
        return relevant_pheromones


    def calculate_pheromone_vector(self, pheromone):
        """
        Calculate the vector contribution of a single pheromone to the agent's movement.
        Uses pre-allocated vectors for efficiency.

        Args:
            pheromone (Pheromone): The pheromone object.

        Returns:
            numpy.ndarray: 2D vector representing pheromone influence [vx, vy].
        """
        # Calculate direction components
        self._temp_vector[0] = np.cos(pheromone.direction)
        self._temp_vector[1] = np.sin(pheromone.direction)
        
        # Scale by strength
        self._temp_vector *= pheromone.strength
        
        if pheromone.type != "Avoidance":
            # Calculate vector from agent to pheromone
            self._direction_vector[0] = pheromone.location[0] - self.pose[0]
            self._direction_vector[1] = pheromone.location[1] - self.pose[1]
            
            # Normalize and scale if non-zero
            norm = np.linalg.norm(self._direction_vector)
            if norm > 1e-6:
                self._direction_vector *= (PHEROMONE_PULL_FACTOR * pheromone.strength / norm)
            else:
                self._direction_vector.fill(0)
            
            # Add the direction vector to the pheromone vector
            self._temp_vector += self._direction_vector
        
        return self._temp_vector.copy()  # Return a copy to prevent modification of the cached vector


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
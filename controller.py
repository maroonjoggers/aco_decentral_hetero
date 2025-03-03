# controller.py
import numpy as np
from environment import *
from agent import *

class Controller:
    def __init__(self, environment):
        """
        Initialize the Controller.

        Args:
            environment (Environment): The environment object.
        """
        self.environment = environment


    def run_step(self):
        """
        Execute one control step for all agents in the environment.
        This includes:
            1. Agent sensing and environment checking.
            2. Pheromone laying and decay.
            3. Pheromone map sharing between neighbors.
            4. Velocity input determination (ACO).
            5. Velocity command application (using Robotarium API - to be integrated in main.py).

        Returns:
            numpy.ndarray: Array of velocity commands (Single Integrator form) for all agents.
        """
        agent_velocities_si = np.zeros((2, len(self.environment.agents))) # Initialize velocities

        for i in range(len(self.environment.agents)):
            agent = self.environment.agents[i]

            # 1. Agent Sensing and Environment Check
            #TODO: Check logic but might be okay
            agent.check_environment(self.environment)           #State change if we found food/goal, as well as handles obstacles (or at least it will)

            # 2. Pheromone Update (Laying own pheromones and decay)
            #TODO: Logic changes as seen in function definition comments
            agent.update_pheromone_map_own(self.environment)    # Lay pheromones based on agent's state and decay existing


        # 3. Pheromone Map Sharing (Decentralized Communication) - AFTER all agents have sensed and potentially laid pheromones in this timestep
        for i in range(len(self.environment.agents)): # Separate loop for sharing after sensing/laying
            agent = self.environment.agents[i]
            neighbors = self.environment.get_agents_within_communication_radius(agent, agent.communication_radius)
            agent.update_pheromone_map_shared(self.environment, neighbors) # Updates own map based on neighbors. TODO: Changes as described within function


        # 4. Determine Velocity Inputs (ACO-based decentralized control) - AFTER pheromone maps are updated
        for i in range(len(self.environment.agents)):
            agent = self.environment.agents[i]
            velocity_input_si = agent.determine_velocity_inputs_aco(self.environment) # ACO velocity calculation    TODO See function
            agent_velocities_si[:, i] = velocity_input_si # Store SI velocity for agent in the returned variable    TODO: CHECK THIS IS IN THE CORRECT FORM



        # 5. Age Update - Increment agent age at each step
        for agent in self.environment.agents:
            agent.update_age()                  #TODO: Check function


        return agent_velocities_si # Return all agents' velocities for Robotarium application in main.py

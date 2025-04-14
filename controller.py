# controller.py
import numpy as np
from environment import *
from agent import *

from stable_baselines3 import SAC
from rl.sac_agent import AgentSAC
from rl.reward import compute_reward
import csv
import os

from network_barriers import *

class Controller:                                                       #TODO: IMPLEMENT ALL NECESSARY FUNCTIONS / ... FOR RL/QP
    def __init__(self, environment):
        """
        Initialize the Controller.

        Args:
            environment (Environment): The environment object.
        """
        self.environment = environment
        self.previous_time = 0.0
        self.ph_plot = None

        num_agents = len(self.environment.agents)

        ############################## RL ##############################

        self.rl_agents = []  # SAC models wrapped
        self.log_files = []
        self.log_writers = []

        self.lambda_update_step_interval = TRAINING_INTERVAL  # Update every 30 steps
        self.current_step = 0
        self.previous_lambda_values = np.ones(NUM_AGENTS) * 0.5 # initial 50/50
    

        # Ensure directories exist
        os.makedirs('logs', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        for i in range(num_agents):
            # Setup logger
            log_file = open(f'logs/agent_{i}_log.csv', mode='w', newline='')
            logger = csv.writer(log_file)
            logger.writerow(['Timestep', 'Lambda', 'Reward'])

            self.log_files.append(log_file)
            self.log_writers.append(logger)

            # Define agent-specific functions
            def get_state_fn(agent_index=i):
                agent = self.environment.agents[agent_index]
                return self.environment.get_state_vector(agent)

            def compute_reward_fn(state, lambda_value):
                return compute_reward(state, lambda_value)

            # Initialize RL agent
            rl_agent = AgentSAC(
                agent_index=i,
                get_state_fn=get_state_fn,
                compute_reward_fn=compute_reward_fn,
                logger=self.log_writers[i],
                training_interval=10
            )

            self.rl_agents.append(rl_agent)

        ############################## RL ##############################

    def run_step(self, current_time):
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
            agent.check_environment(self.environment)           #State change if we found food/goal, as well as handles obstacles

        if current_time - self.previous_time >= PH_LAYING_RATE:
            for i in range(len(self.environment.agents)):
                # 2. Pheromone Update (Laying own pheromones and decay)
                agent = self.environment.agents[i]
                # Check for obstacles and lay avoidance pheromones if needed
                nearby_obstacles, obstacle_angle = self.environment.get_nearby_obstacles(agent.pose[:2], agent.sensing_radius)
                nearby_hazards = self.environment.get_nearby_hazards(agent.pose[:2], agent.sensing_radius)
                if nearby_obstacles or nearby_hazards:
                    agent.update_pheromone_map_own(self.environment, avoidance=True, avoidance_angle=obstacle_angle)
                else:
                    agent.update_pheromone_map_own(self.environment)    # Lay pheromones based on agent's state and decay existing
                self.previous_time = current_time
            
            # Plot all pheromones
            if self.ph_plot:
                self.ph_plot.remove()
            
            pheromone_colors = {
                "Return Home": "r",
                "To Food": "k",
                "Avoidance": "m" 
            }
    
            # Get all pheromones from the dictionary
            all_pheromones = list(self.environment.pheromone_dict.values())
            xVals = [ph.location[0] for ph in all_pheromones]
            yVals = [ph.location[1] for ph in all_pheromones]
            colors = [pheromone_colors[ph.type] for ph in all_pheromones]
            strengths = [min(1.0, ph.strength) for ph in all_pheromones]
            self.ph_plot = self.environment.robotarium.axes.scatter(xVals, yVals, s=15, c=colors, alpha=strengths)


        # 3. Pheromone Map Sharing (Decentralized Communication) - AFTER all agents have sensed and potentially laid pheromones in this timestep
        for i in range(len(self.environment.agents)): # Separate loop for sharing after sensing/laying
            agent = self.environment.agents[i]
            neighbors = self.environment.get_agents_within_communication_radius(agent, agent.communication_radius)
            agent.update_pheromone_map_shared(self.environment, neighbors) # Updates own map based on neighbors

        # 4. Determine Velocity Inputs (ACO-based decentralized control) - AFTER pheromone maps are updated
        for i in range(len(self.environment.agents)):
            agent = self.environment.agents[i]
            velocity_input_si = agent.determine_velocity_inputs_aco(self.environment, current_time) # ACO velocity calculation
            agent_velocities_si[:, i] = velocity_input_si # Store SI velocity for agent in the returned variable

        # 5. Age Update - Increment agent age at each step
        for agent in self.environment.agents:
            agent.update_age()

        return agent_velocities_si # Return all agents' velocities for Robotarium application in main.py

    def close(self):
        """
        Close log files and save models for all RL agents.
        """
        for i, rl_agent in enumerate(self.rl_agents):
            rl_agent.save()
            self.log_files[i].close()
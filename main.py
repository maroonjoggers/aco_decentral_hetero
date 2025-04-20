# main.py
import cProfile
import pstats
import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
import time
import matplotlib.patches as patches

from utils import * # Import configuration parameters and agent trait profiles
from environment import Environment
from controller import Controller
from network_barriers import *
from rl.plot_lambda import plot_all_agents

import os
import csv

def main():
    # --- 1. Robotarium Initialization ---
    # Parameters from utils.py
    initalConditions = determineInitalConditions()
    r = robotarium.Robotarium(number_of_robots=NUM_AGENTS, show_figure=PLOTTING, initial_conditions=initalConditions, sim_in_real_time=True)
    r.time_step = 0.05
    #TODO: Notes on above line initialization
    # 1. See notes on NUM_AGENTS in utily.py --> Needs to be max total possible alive at any one point
    # 2. Initial Conditions should not be zero, they should be the spawn point. It would probably be better to make a simple formation because they can't all literally be in the same spot physically at one time

    # Instantiate SI to Uni dynamics and SI position controller (for potential leader or centralized elements later)
    si_to_uni_dyn = create_si_to_uni_dynamics() # or create_si_to_uni_dynamics_with_backwards_motion() if needed
    si_position_controller = create_si_position_controller() # Example controller - not directly used in decentralized ACO


    # --- 2. Barrier Certificates (for collision avoidance) ---
    # Barrier certificate for single integrator dynamics - adjust parameters as needed
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(barrier_gain=100, safety_radius=0.2) # Example parameters

    # --- 3. Initialize Environment and Controller ---
    # Instantiate Environment with parameters from utils.py
    env = Environment(
        boundary_points=ROBOTARIUM_BOUNDARIES,
        home_location=HOME_LOCATION,
        food_locations=FOOD_LOCATIONS,
        obstacle_locations=OBSTACLE_LOCATIONS,
        hazard_locations=HAZARD_LOCATIONS,
        num_agents=NUM_AGENTS,
        agent_traits_profiles=AGENT_TRAIT_PROFILES,
        agent_ICs = initalConditions,
        robotarium = r
    )
    #TODO: The above line will run the self.initialize_agents() function in the init of the Environment() class, see that

    # Instantiate Controller, passing the Environment object
    controller = Controller(env)

    """RL Episodes Tracking"""
    episode_counter_path = "models/episode_counter.txt"
    start_episode = 1
    if os.path.exists(episode_counter_path) and USE_CHECKPOINT:
        with open(episode_counter_path, "r") as f:
            start_episode = int(f.read()) + 1

    print(f"=== Starting Episode {start_episode} ===")

    episode_tasks_path = "logs/tasks_completed_log.csv"
    os.makedirs("logs", exist_ok=True)

    if not USE_CHECKPOINT and os.path.exists(episode_tasks_path):
        open(episode_tasks_path, 'w').close()  # Clear the file

    episode_log_file = open(episode_tasks_path, mode='a', newline='')
    episode_logger = csv.writer(episode_log_file)

    # Write header if file doesn't exist yet
    if not os.path.exists(episode_tasks_path):
        episode_logger.writerow(["Episode", "Tasks_Completed"])


    # --- 4. Initialize Visualization ---
    # Use Robotarium's figure and axes
    ax = r.axes
    ax.set_xlim(ROBOTARIUM_BOUNDARIES[0], ROBOTARIUM_BOUNDARIES[1])
    ax.set_ylim(ROBOTARIUM_BOUNDARIES[2], ROBOTARIUM_BOUNDARIES[3])

    # Plot home and food locations
    home_scatter = ax.scatter(HOME_LOCATION[0], HOME_LOCATION[1], s=100, c='b', zorder=1)
    xVals = [location[0] for location in FOOD_LOCATIONS]
    yVals = [location[1] for location in FOOD_LOCATIONS]
    food_scatter = ax.scatter(xVals, yVals, s=100, c='g', zorder=1)

    #Plot Obstacles
    for obstacle in OBSTACLE_LOCATIONS:
        if obstacle["shape"] == "rectangle":
            center_x, center_y = obstacle["center"]
            width = obstacle["width"]
            height = obstacle["height"]
            rect = patches.Rectangle((center_x - width / 2, center_y - height / 2), width, height, linewidth=1, edgecolor='r', facecolor='r', alpha=0.5)
            r.axes.add_patch(rect)
        elif obstacle["shape"] == "circle":
            center_x, center_y = obstacle["center"]
            radius = obstacle["radius"]
            circle = patches.Circle((center_x, center_y), radius)
            r.axes.add_patch(circle)


    # Initialize scatter plots for visualization elements with appropriate zorder
    circles_scatter = ax.scatter([], [], s=2, c='y', zorder=2)
    edges_scatter = ax.scatter([], [], s=2, c='c', zorder=2)
    arrow_scatter = ax.scatter([], [], s=2, c='m', zorder=2)

    def update_visualizations():
        # Update circles for sensing radii
        circles_data = []
        for agent in env.agents:
            center = agent.pose[:2]
            radius = agent.sensing_radius
            theta = np.linspace(0, 2*np.pi, 30)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            circles_data.extend(list(zip(x, y)))
        if circles_data:
            circles_scatter.set_offsets(circles_data)
        else:
            circles_scatter.set_offsets(np.empty((0, 2)))

        # Update edges for communication
        edges_data = []
        for agent in env.agents:
            agent_pos = agent.pose[:2]
            for neighbor in env.get_agents_within_communication_radius(agent, agent.communication_radius):
                neighbor_pos = neighbor.pose[:2]
                x = np.linspace(agent_pos[0], neighbor_pos[0], 20)
                y = np.linspace(agent_pos[1], neighbor_pos[1], 20)
                edges_data.extend(list(zip(x, y)))
        if edges_data:
            edges_scatter.set_offsets(edges_data)
        else:
            edges_scatter.set_offsets(np.empty((0, 2)))

        # Update arrows for velocities
        arrow_data = []
        for agent in env.agents:
            agent_pos = agent.pose[:2]
            velocity_norm = np.linalg.norm(agent.velocity_vector)
            if velocity_norm > 0:  # Only normalize if velocity is non-zero
                velocity_vector = agent.velocity_vector / (4 * velocity_norm)
                x = np.linspace(agent_pos[0], agent_pos[0] + velocity_vector[0], 10)
                y = np.linspace(agent_pos[1], agent_pos[1] + velocity_vector[1], 10)
                arrow_data.extend(list(zip(x, y)))
        if arrow_data:
            arrow_scatter.set_offsets(arrow_data)
        else:
            arrow_scatter.set_offsets(np.empty((0, 2)))

    start_time = time.time()
    last_time = time.time()  # Track the time of the last iteration
    last_print_time = time.time()  # Track when we last printed stats
    iteration_times = []  # Store iteration times for averaging
    while True:
        # Calculate time since last iteration
        current_iteration_time = time.time() - last_time
        iteration_times.append(current_iteration_time)
        
        # Print average every second
        if time.time() - last_print_time >= 1.0:
            avg_iteration_time = sum(iteration_times) / len(iteration_times)
            print(f"\nAverage loop iteration time: {avg_iteration_time:.6f} seconds")
            print(f"Elapsed time: {int(time.time() - start_time)} seconds")
            iteration_times = []  # Reset the list
            last_print_time = time.time()
            
        last_time = time.time()  # Update last_time for next iteration

        current_time = time.time() - start_time  # Total elapsed time

        # need to get states and apply them
        x = r.get_poses()
        env.update_poses(x)
        env.needs_agent_tree_update = True  # Mark agent tree for update

        # a) Run Controller Step - Decentralized ACO velocity calculation
        agent_velocities_si_nominal = controller.run_step(current_time) # TODO: This is where the largest chunk of our actual algorithm functionality lies

        # b.1) Apply NETWORK barriers for staying in communication ............
        if not WITH_LAMBDA:
            agent_velocities_si = network_barriers_with_obstacles_final(agent_velocities_si_nominal, x[:2], env, None, False)
        else:
            lambda_values = np.array([
                controller.rl_agents[i].select_lambda(current_time)
                for i in range(NUM_AGENTS)
            ])
            agent_velocities_si = network_barriers_with_obstacles_final(agent_velocities_si_nominal, x[:2], env, lambda_values, True)

        #agent_velocities_si = network_barriers_obstacles3(agent_velocities_si_nominal, x[:2], env, lam)

        # b.2) Apply SAFETY Barrier Certificates - Ensure safety (collision avoidance, boundary constraints)
        # safe_velocities_si = si_barrier_cert(agent_velocities_si, x[:2]) # Barrier certificate application
        safe_velocities_si = agent_velocities_si

        env.update_velocities(safe_velocities_si)

        # c) Convert SI velocities to Unicycle Velocities (Robotarium-compatible)
        agent_velocities_uni = si_to_uni_dyn(safe_velocities_si, x) # SI to Uni velocity transformation

        # d) Set Velocities in Robotarium - Command robots to move
        r.set_velocities(np.arange(NUM_AGENTS), agent_velocities_uni)

        # Update visualizations
        update_visualizations()

        # e) Iterate Robotarium Simulation - Step the simulation forward
        r.step()

        if current_time >= MAX_TIME:
            break

    print("YAY! TASKS COMPLETED: " + str(env.tasks_completed))
    episode_logger.writerow([start_episode, env.tasks_completed])
    episode_log_file.close()

    # --- 6. Experiment End, RL plots, Saving, and Cleanup ---
    print(f"=== Episode {start_episode} complete! ===")
    os.makedirs("models", exist_ok=True)
    with open(episode_counter_path, "w") as f:
        f.write(str(start_episode))

    controller.close()
    if PLOT_LAMBDA: plot_all_agents(NUM_AGENTS)
    r.call_at_scripts_end() # Robotarium cleanup and display message


if __name__ == "__main__":
    # Create a profiler
    pr = cProfile.Profile()
    
    # Start profiling
    pr.enable()
    
    # Run the main function
    main()
    
    # Stop profiling
    pr.disable()
    
    # Save the profiling results to a file
    if PLOTTING:
        text = "profiling_stats.txt"
    else:
        text = "profiling_stats_no_figure.txt"
    with open(text, 'w') as f:
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats('cumulative')
        ps.print_stats()

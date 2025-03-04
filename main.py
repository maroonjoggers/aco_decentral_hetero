# main.py
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *
import time

from utils import * # Import configuration parameters and agent trait profiles
from environment import Environment
from controller import Controller

def plot_home_and_food():

    #Plot home location
    r.axes.scatter(HOME_LOCATION[0], HOME_LOCATION[1], s=100, c='b')

    #Plot food locations
    xVals = [location[0] for location in FOOD_LOCATIONS]
    yVals = [location[1] for location in FOOD_LOCATIONS]
    r.axes.scatter(xVals, yVals, s=100, c='g')


# --- 1. Robotarium Initialization ---
# Parameters from utils.py
initalConditions = determineInitalConditions()
r = robotarium.Robotarium(number_of_robots=NUM_AGENTS, show_figure=True, initial_conditions=initalConditions, sim_in_real_time=True)

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


# --- 4. Set Initial Poses in Robotarium ---
#TODO: This is all a mess. There is no such thing as r.set_poses. The below 2 lines can basically just go away. We don't need a separate function to get poses because that's part of the robotarium functionality
# - What we may or may not need is a function to take what the robotarium gives us as the pose info and transform it for our uses

#initial_poses = env.get_agent_poses() # Get initial poses from Environment (randomly initialized)
#r.set_poses(initial_poses) # Set initial poses in Robotarium


# --- 5. Run Simulation Loop ---
#TODO: Do we want to run this based on a number of iterations? I think not
# 1) We need to define the end state for our experiment. Could be multiple:
#  a) Reached a certain time
#  b) Reached a certain population min/max
# 2) PS: each robotarium iteration is 0.033 seconds

#TODO: Pose Handling
# 1) The get_agent_poses(), as stated previously, doesn't make sense and can be replaced with robotarium's built in functionality
# 2) The only issue we may face is matching which agent we are getting the robotarium's info about and corresponding that to our info about each agent. This shouldn't be too crazy

plot_home_and_food()

start_time = time.time()
while True:
    current_time = time.time() - start_time
    print(current_time)


    # need to get states and apply them
    x = r.get_poses()

    env.updatePoses(x)

    # a) Run Controller Step - Decentralized ACO velocity calculation
    agent_velocities_si = controller.run_step() # TODO: This is where the largest chunk of our actual algorithm functionality lies

    # b) Apply Barrier Certificates - Ensure safety (collision avoidance, boundary constraints)
    #safe_velocities_si = si_barrier_cert(agent_velocities_si, env.get_agent_poses()[:2,:]) # Barrier certificate application
    safe_velocities_si = si_barrier_cert(agent_velocities_si, x[:2]) # Barrier certificate application

    # c) Convert SI velocities to Unicycle Velocities (Robotarium-compatible)
    #agent_velocities_uni = si_to_uni_dyn(safe_velocities_si, env.get_agent_poses()) # SI to Uni velocity transformation
    agent_velocities_uni = si_to_uni_dyn(safe_velocities_si, x) # SI to Uni velocity transformation

    # d) Set Velocities in Robotarium - Command robots to move
    r.set_velocities(np.arange(NUM_AGENTS), agent_velocities_uni)

    # e) Iterate Robotarium Simulation - Step the simulation forward
    r.step()

    if current_time >= MAX_TIME:
        break


# --- 6. Experiment End and Cleanup ---
r.call_at_scripts_end() # Robotarium cleanup and display message
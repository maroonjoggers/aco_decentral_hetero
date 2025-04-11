import csv
import matplotlib.pyplot as plt
import os
from datetime import datetime
from utils import NUM_AGENTS

os.makedirs('plots', exist_ok=True)

def read_agent_log(agent_index, log_folder='logs'):
    log_file = os.path.join(log_folder, f'agent_{agent_index}_log.csv')

    if not os.path.exists(log_file):
        print(f"[Warning] Log file not found for agent {agent_index}")
        return None

    timesteps, lambdas, rewards = [], [], []

    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            timesteps.append(float(row[0]))
            lambdas.append(float(row[1]))
            rewards.append(float(row[2]))

    return timesteps, lambdas, rewards

def save_plot(fig, filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join('plots', f'{filename}_{timestamp}.png')
    fig.savefig(filepath)
    print(f"[Saved] {filepath}")

def plot_combined_agents(agent_data, metric='lambda'):
    fig = plt.figure(figsize=(8, 5))

    for agent_index, data in agent_data.items():
        timesteps, lambdas, rewards = data
        values = lambdas if metric == 'lambda' else rewards
        plt.plot(timesteps, values, label=f'Agent {agent_index}')

    plt.xlabel('Timestep')
    ylabel = 'Lambda Value' if metric == 'lambda' else 'Reward'
    plt.ylabel(ylabel)
    plt.title(f'Combined Agents {ylabel} Evolution')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    save_plot(fig, f'combined_{metric}')

    plt.show()


def plot_all_agents(num_agents=NUM_AGENTS, log_folder='logs'):
    agent_data = {}

    for i in range(num_agents):
        data = read_agent_log(i, log_folder=log_folder)
        if data:
            agent_data[i] = data
            # Optional: plot individual agents
            # plot_individual_agent(i, *data)

    if agent_data:
        plot_combined_agents(agent_data, metric='lambda')
        plot_combined_agents(agent_data, metric='reward')
    else:
        print("No agent logs found!")

if __name__ == "__main__":
    plot_all_agents()

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

    timesteps, lambdas, rewards, cumulative_rewards = [], [], [], []

    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            timesteps.append(float(row[0]))
            lambdas.append(float(row[1]))
            rewards.append(float(row[2]))
            # Handle empty cumulative reward values (logged as empty string)
            cumulative_rewards.append(float(row[3]) if row[3] != "" else None)

    return timesteps, lambdas, rewards, cumulative_rewards

def save_plot(fig, filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join('plots', f'{filename}_{timestamp}.png')
    fig.savefig(filepath)
    print(f"[Saved] {filepath}")

def plot_combined_agents(agent_data, metric='lambda'):
    fig = plt.figure(figsize=(8, 5))

    for agent_index, data in agent_data.items():
        timesteps, lambdas, rewards, cumulative_rewards = data

        if metric == 'lambda':
            values = lambdas
            ylabel = 'Lambda Value'
        elif metric == 'reward':
            values = rewards
            ylabel = 'Reward'
        elif metric == 'cumulative_reward':
            # Filter out None values (since cumulative reward is only logged at episode end)
            values = [cr for cr in cumulative_rewards if cr is not None]
            timesteps = [t for t, cr in zip(timesteps, cumulative_rewards) if cr is not None]
            ylabel = 'Cumulative Reward'
        else:
            raise ValueError(f"Unknown metric: {metric}")

        plt.plot(timesteps, values, label=f'Agent {agent_index}')

    plt.xlabel('Timestep')
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
        plot_combined_agents(agent_data, metric='cumulative_reward')
    else:
        print("No agent logs found!")

if __name__ == "__main__":
    plot_all_agents()

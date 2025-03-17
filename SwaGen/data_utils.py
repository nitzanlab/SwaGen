import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def generate_agent_positions(num_nodes, mean_loc=500, var_loc=500, mean_range=100, var_range=100):
    mean_x = np.random.normal(loc=mean_loc, scale=np.sqrt(var_loc))
    mean_y = np.random.normal(loc=mean_loc, scale=np.sqrt(var_loc))
    x_range = np.random.normal(loc=mean_range, scale=np.sqrt(var_range))
    y_range = np.random.normal(loc=mean_range, scale=np.sqrt(var_range))
    x_positions = np.random.uniform(mean_x - x_range / 2, mean_x + x_range / 2, num_nodes)
    y_positions = np.random.uniform(mean_y - y_range / 2, mean_y + y_range / 2, num_nodes)
    x_positions = np.clip(x_positions, 0, 1000)
    y_positions = np.clip(y_positions, 0, 1000)
    return np.column_stack((x_positions, y_positions))

def add_graph_features(data, max_distance):
    positions = data.x.cpu().numpy()
    center_of_mass = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    num_neighbors = (distances < max_distance).sum(axis=1) - 1
    vector_to_com = positions - center_of_mass
    distance_to_com = np.linalg.norm(vector_to_com, axis=1)
    angle_to_com = np.arctan2(vector_to_com[:, 1], vector_to_com[:, 0])
    features = np.column_stack((positions[:, 0], positions[:, 1], num_neighbors, distance_to_com, angle_to_com))
    data.x = torch.tensor(features, dtype=torch.float)
    return data

def visualize_swarm(positions, title="Swarm Visualization", save_path=None):
    positions = positions.cpu().detach().numpy()
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(positions[:, 0], positions[:, 1], c='blue', edgecolors='black', s=50)
    plt.colorbar(scatter, label='Row Index')
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

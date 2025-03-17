import numpy as np
import torch
import matplotlib.pyplot as plt
import threading
from torch_geometric.data import Data
from data_utils import generate_agent_positions, add_graph_features

def create_graph_from_pos(positions, max_distance=50):
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    W = (distances < max_distance).astype(float)
    np.fill_diagonal(W, 0)
    edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
    x = torch.tensor(positions, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.original_positions = torch.tensor(positions, dtype=torch.float)
    data = add_graph_features(data, max_distance)
    return data

def arrange_agents_in_arrow_lines(N, width, height):
    points = []
    num_points_per_line = N // 2
    x_spacing = width / (num_points_per_line + 1)
    y_spacing = height / (num_points_per_line + 1)
    for i in range(num_points_per_line):
        points.append((i * x_spacing + x_spacing, 0 + y_spacing))
    for i in range(num_points_per_line):
        points.append((0 + x_spacing, i * y_spacing + y_spacing))
    points = np.array(points)
    theta = np.radians(180)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_points = points @ R.T
    min_x, min_y = rotated_points.min(axis=0)
    max_x, max_y = rotated_points.max(axis=0)
    scale = min(width / (max_x - min_x), height / (max_y - min_y))
    return (rotated_points - [min_x, min_y]) * scale

def arrange_agents_in_arrow(N, width, height):
    points = []
    num_rows = int(np.ceil((np.sqrt(8 * N + 1) - 1) / 2))
    x_spacing = width / (2 * num_rows + 1)
    y_spacing = height / (2 * num_rows + 1)
    index = 0
    for row in range(num_rows):
        for col in range(row + 1):
            if index >= N:
                break
            points.append((col * x_spacing, row * y_spacing))
            index += 1
        if index >= N:
            break
    points = np.array(points)
    points[:, 0] += (width / 2 - num_rows * x_spacing / 2)
    points[:, 1] += (height / 2 - num_rows * y_spacing / 2)
    theta = np.radians(270)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_points = points @ R.T
    min_x, min_y = rotated_points.min(axis=0)
    max_x, max_y = rotated_points.max(axis=0)
    scale = min(width / (max_x - min_x), height / (max_y - min_y))
    return (rotated_points - [min_x, min_y]) * scale

# Additional helper functions for agent-based simulation
def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def angular_deviation(points, preferred_direction):
    preferred_direction = preferred_direction / np.linalg.norm(preferred_direction)
    angles = np.arctan2(points[:, 1], points[:, 0]) - np.arctan2(preferred_direction[1], preferred_direction[0])
    angles = normalize_angle(angles)
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    deviations = normalize_angle(angles - mean_angle)
    return (deviations + np.pi) / (2 * np.pi)

def knn_graph(points, k=10, max_distance=50):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    n_points = points.shape[0]
    adjacency_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(k):
            if distances[i, j] <= max_distance:
                adjacency_matrix[i, indices[i, j]] = 1
    return adjacency_matrix

def dfs(node, visited, W):
    stack = [node]
    while stack:
        current = stack.pop()
        for neighbor, weight in enumerate(W[current]):
            if weight != 0 and not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)

def count_connected_components(W):
    n = len(W)
    visited = [False] * n
    component_count = 0
    for node in range(n):
        if not visited[node]:
            component_count += 1
            visited[node] = True
            dfs(node, visited, W)
    return component_count

class Agent:
    def __init__(self, x, y, vx, vy, informed=False, predator=False):
        self.position = np.array([x, y])
        self.velocity = np.array([vx, vy])
        self.informed = informed
        self.predator = predator
        if informed:
            self.predator = False

def calculate_distance(pos1, pos2, width, height):
    delta_pos = np.abs(pos1 - pos2)
    delta_pos[0] = min(delta_pos[0], width - delta_pos[0])
    delta_pos[1] = min(delta_pos[1], height - delta_pos[1])
    return np.linalg.norm(delta_pos, 2)

def wrap_distance(a, b, max_bound):
    dist = abs(a - b)
    return min(dist, max_bound - dist)

def wrap_distance_array(a, b, max_bound):
    dist = np.abs(a - b)
    return np.minimum(dist, max_bound - dist)

def calculate_norm(delta_pos):
    return np.abs(delta_pos).sum()

def adjust_direction(agent_pos, other_pos, width, height):
    delta_pos = wrap_distance_array(agent_pos, other_pos, np.array([width, height]))
    if abs(agent_pos[0] - other_pos[0]) > width / 2:
        delta_pos[0] = np.sign(agent_pos[0] - other_pos[0]) * delta_pos[0]
    else:
        delta_pos[0] = np.sign(other_pos[0] - agent_pos[0]) * delta_pos[0]
    if abs(agent_pos[1] - other_pos[1]) > height / 2:
        delta_pos[1] = np.sign(agent_pos[1] - other_pos[1]) * delta_pos[1]
    else:
        delta_pos[1] = np.sign(other_pos[1] - agent_pos[1]) * delta_pos[1]
    return delta_pos / calculate_norm(delta_pos)

def calculate_desired_direction(agent_pos, agent_vel, j, agents_pos, agents_vel,
                                informed, predator, live_array, agents_predator,
                                min_dist, interaction_range, weighting_term,
                                killing_range, preferred_direction, width, height):
    desired_direction = np.zeros(2)
    closest_distance = float('inf')
    closest_neighbor = None
    for i, (other_pos, other_vel) in enumerate(zip(agents_pos, agents_vel)):
        if not live_array[i]:
            continue
        if not np.array_equal(other_pos, agent_pos):
            distance = calculate_distance(agent_pos, other_pos, width, height)
            if predator:
                if (distance < closest_distance) and (distance < interaction_range) and (not agents_predator[i]):
                    closest_distance = distance
                    closest_neighbor = other_pos
                continue
            if agents_predator[i]:
                if distance < killing_range:
                    live_array[j] = False
                    break
                elif distance < interaction_range:
                    desired_direction += -adjust_direction(agent_pos, other_pos, width, height)
                    continue
            if distance < min_dist:
                desired_direction += -adjust_direction(agent_pos, other_pos, width, height)
            elif distance < interaction_range:
                desired_direction += 0.1 * adjust_direction(agent_pos, other_pos, width, height)
                if np.linalg.norm(other_vel, 1) > 0:
                    desired_direction += other_vel / np.linalg.norm(other_vel, 1)
    if predator and closest_neighbor is not None:
        desired_direction = adjust_direction(agent_pos, closest_neighbor, width, height)
    if np.linalg.norm(desired_direction) > 0:
        desired_direction = desired_direction / np.linalg.norm(desired_direction)
    if informed:
        desired_direction = (1 - weighting_term) * desired_direction + weighting_term * preferred_direction
        desired_direction = desired_direction / np.linalg.norm(desired_direction)
    return desired_direction

def update_agents(agents_pos, agents_vel, live_array, agents_informed,
                  agents_predator, min_dist, interaction_range,
                  weighting_term, killing_range, preferred_direction,
                  speed, width, height):
    new_agents_pos = np.copy(agents_pos)
    new_agents_vel = np.copy(agents_vel)
    for i in range(len(agents_pos)):
        desired_direction = calculate_desired_direction(agents_pos[i], agents_vel[i], i,
                                                        agents_pos, agents_vel,
                                                        agents_informed,
                                                        agents_predator, live_array,
                                                        agents_predator, min_dist,
                                                        interaction_range, weighting_term,
                                                        killing_range, preferred_direction,
                                                        width, height)
        new_agents_vel[i] = desired_direction * speed * (1.5 if agents_predator[i] else 1.0)
        new_agents_pos[i] += new_agents_vel[i]
        new_agents_pos[i][0] %= width
        new_agents_pos[i][1] %= height
    return new_agents_pos, new_agents_vel

def run_single_simulation(res_array_naive, res_array_predicted, model, simulation_index):
    N = 52
    width, height = 1000, 1000
    min_dist = 5
    interaction_range = 50
    speed = 2
    informed_fraction = 0.2
    predator_fraction = 0.025
    killing_range = 5
    preferred_direction = np.array([0.5, 0.5])
    weighting_term = 0.5
    num_frames = 1000

    agents_pos = generate_agent_positions(N)
    agents_vel = np.random.randn(N, 2)
    agents_vel = (agents_vel.T / np.linalg.norm(agents_vel, axis=1) * speed).T
    agents_informed = np.random.rand(N) < informed_fraction
    agents_predator = np.random.rand(N) < predator_fraction
    agents_predator[agents_informed] = False
    agents_predator[:] = False
    agents_predator[N - 1] = True
    agents_informed[N - 1] = False
    agents_pos[N - 1, :] = np.random.rand(1, 2) * np.array([width, height])
    agents_predator[N - 2] = True
    agents_informed[N - 2] = False
    agents_pos[N - 2, :] = np.random.rand(1, 2) * np.array([width, height])
    live_array = np.ones(N).astype(int)

    for frame_num in range(num_frames):
        print(f"Simulation {simulation_index + 1} - Frame {frame_num + 1}")
        agents_pos, agents_vel = update_agents(agents_pos, agents_vel, live_array,
                                               agents_informed, agents_predator,
                                               min_dist, interaction_range,
                                               weighting_term, killing_range,
                                               preferred_direction, speed, width, height)
        live_array_bool = live_array.astype(bool) & (~agents_predator)
        live_agents_pos = agents_pos[live_array_bool, :]
        print(f"Alive agents: {np.sum(live_array) - 2}")

    print(f"Simulation {simulation_index + 1} completed.")
    res_array_naive.append(np.sum(live_array) - 2)

def run_simulations(res_array_naive, res_array_predicted, model):
    threads = []
    for i in range(1000):
        thread = threading.Thread(target=run_single_simulation, args=(res_array_naive, res_array_predicted, model, i))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    res_array_naive = []
    res_array_predicted = []
    run_simulations(res_array_naive, res_array_predicted, model=1)
    print("Naive Results:", res_array_naive)
    print("Predicted Results:", res_array_predicted)
    print("Mean:", np.mean(res_array_naive))
    print("Variance:", np.var(res_array_naive))

import torch
import torch.nn as nn

class DomainOfDangerLoss(nn.Module):
    def __init__(self, r, grid_size):
        super(DomainOfDangerLoss, self).__init__()
        self.r = r
        self.grid_size = grid_size

    def distance(self, xx, yy, px, py):
        return torch.sqrt((xx - px).pow(2) + (yy - py).pow(2) + 1e-6)

    def forward(self, positions):
        width, height = self.grid_size, self.grid_size
        num_nodes = positions.shape[0]
        danger_zone = torch.zeros((width, height), dtype=torch.float32, device=positions.device)

        x = torch.arange(0, width, device=positions.device)
        y = torch.arange(0, height, device=positions.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        xx, yy = xx.float(), yy.float()

        for i in range(num_nodes):
            px, py = positions[i]
            dist = self.distance(xx, yy, px, py)
            danger_zone += torch.sigmoid(-1 * (dist - self.r))

        return torch.mean(torch.sigmoid(danger_zone))

class MinimalChangeLoss(nn.Module):
    def __init__(self):
        super(MinimalChangeLoss, self).__init__()
        self.mse_loss_fn = nn.MSELoss()

    def forward(self, original_positions, optimized_positions):
        spatial_original = original_positions[:, :2]
        return self.mse_loss_fn(spatial_original, optimized_positions)

class DiffusionLoss(nn.Module):
    def __init__(self, taus=[5.0, 10.0], max_distance=50):
        super(DiffusionLoss, self).__init__()
        self.taus = taus
        self.max_distance = max_distance

    def forward(self, optimized_positions):
        num_nodes = optimized_positions.shape[0]
        distances = torch.cdist(optimized_positions, optimized_positions)
        adjacency_matrix = torch.sigmoid(-1 * (1 / self.max_distance) * (distances - self.max_distance))
        adjacency_matrix = adjacency_matrix - torch.diag(torch.diag(adjacency_matrix))

        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        laplacian = degree_matrix - adjacency_matrix
        degree_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(adjacency_matrix, dim=1) + 1e-6))
        normalized_laplacian = torch.mm(degree_inv_sqrt, torch.mm(laplacian, degree_inv_sqrt))

        total_loss = 0.0
        for tau in self.taus:
            heat_filter = torch.matrix_exp(-tau * normalized_laplacian)
            for i in range(num_nodes):
                delta_signal = torch.zeros(num_nodes, device=optimized_positions.device, dtype=optimized_positions.dtype)
                delta_signal = delta_signal.scatter(0, torch.tensor([i % num_nodes], device=optimized_positions.device), 1.0)
                smoothed_signal = torch.mv(heat_filter, delta_signal)
                coeff_var = torch.std(smoothed_signal) / (torch.mean(smoothed_signal) + 1e-6)
                total_loss += coeff_var

        return total_loss / (num_nodes * len(self.taus))

class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()

    def forward(self, mean, log_var):
        return -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

class RepulsionLoss(nn.Module):
    def __init__(self, min_distance=5.0):
        super(RepulsionLoss, self).__init__()
        self.min_distance = min_distance

    def forward(self, positions):
        distances = torch.cdist(positions, positions)
        mask = torch.eye(distances.size(0), device=distances.device).bool()
        distances = distances.masked_fill(mask, float('inf'))
        violations = torch.relu(self.min_distance - distances)
        return violations.sum() / (positions.size(0) ** 2)

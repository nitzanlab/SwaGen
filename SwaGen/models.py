import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GenerativeGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(GenerativeGNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim // 2, hidden_dim // 2)
        self.conv4 = SAGEConv(hidden_dim // 2, hidden_dim // 2)
        self.linear = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x, edge_index):
        x = x / 1000 - 0.5
        x = torch.sin(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        mean = x[:, :x.shape[1] // 2]
        log_var = x[:, (x.shape[1] // 2):]
        epsilon = torch.randn_like(log_var).to(x.device)
        z = mean + torch.exp(log_var) * epsilon
        x = torch.relu(self.conv3(z, edge_index))
        x = torch.relu(self.conv4(x, edge_index))
        return torch.sigmoid(self.linear(x)) * 1000, mean, log_var

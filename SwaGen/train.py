import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Data
from models import GenerativeGNN
from losses import DomainOfDangerLoss, MinimalChangeLoss, DiffusionLoss, KLD, RepulsionLoss
from data_utils import generate_agent_positions, add_graph_features, visualize_swarm
import numpy as np

def train_gnn(model, data_loader, optimizer, domain_loss_fn,
              minimal_loss_fn, diffusion_loss_fn, kld_loss_fn, repulsion_loss_fn,
              epochs, device, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, epsilon=1.0, tau=0.9,
              save_path='model_checkpoint.pth'):
    mean_losses = {'dod': [], 'mc': [], 'diff': [], 'KLD': [], 'repulse': []}
    model.to(device)
    curr_alpha = alpha
    for epoch in range(epochs):
        epoch_dod_loss = epoch_mc_loss = epoch_diff_loss = epoch_kld_loss = epoch_repulse_loss = 0.0
        model.train()
        total_loss = 0.0
        for data in data_loader:
            optimizer.zero_grad()
            data = data.to(device)
            original_positions = data.original_positions
            edge_index = data.edge_index
            optimized_positions, mean, log_var = model(data.x, edge_index)
            dod_loss = domain_loss_fn(optimized_positions)
            mc_loss = minimal_loss_fn(original_positions, optimized_positions)
            diff_loss = diffusion_loss_fn(optimized_positions)
            kld_loss = kld_loss_fn(mean, log_var)
            repulse_loss = repulsion_loss_fn(optimized_positions)

            epoch_dod_loss += dod_loss.item()
            epoch_mc_loss += mc_loss.item()
            epoch_diff_loss += diff_loss.item()
            epoch_kld_loss += kld_loss.item()
            epoch_repulse_loss += repulse_loss.item()

            loss = curr_alpha * mc_loss + beta * dod_loss - gamma * diff_loss + delta * kld_loss + epsilon * repulse_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_losses['dod'].append(epoch_dod_loss / len(data_loader))
        mean_losses['mc'].append(epoch_mc_loss / len(data_loader))
        mean_losses['diff'].append(epoch_diff_loss / len(data_loader))
        mean_losses['KLD'].append(epoch_kld_loss / len(data_loader))
        mean_losses['repulse'].append(epoch_repulse_loss / len(data_loader))
        curr_alpha = alpha * (tau ** epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        print(f"  Mean Domain of Danger Loss: {mean_losses['dod'][-1]:.4f}")
        print(f"  Mean Minimal Change Loss: {mean_losses['mc'][-1]:.4f}")
        print(f"  Mean Diffusion Loss: {mean_losses['diff'][-1]:.4f}")
        print(f"  Mean KLD Loss: {mean_losses['KLD'][-1]:.4f}")
        print(f"  Mean Repulse Loss: {mean_losses['repulse'][-1]:.4f}")
        visualize_swarm(optimized_positions, title=f"Swarm at Epoch {epoch + 1}")

    # Plot loss trends
    for key, label in zip(mean_losses, ['Domain of Danger', 'Minimal Change', 'Diffusion', 'KLD', 'Repulse']):
        plt.figure(figsize=(10, 6))
        plt.plot(mean_losses[key], label=f'{label} Loss')
        plt.xlabel('Epoch')
        plt.ylabel(f'Mean {label} Loss')
        plt.title(f'{label} Loss Trend')
        plt.grid(True)
        plt.legend()
        plt.show()

    torch.save(model.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")
    return model

def test_gnn(model, data_loader, device, domain_loss_fn, minimal_loss_fn, diffusion_loss_fn):
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            original_positions = data.original_positions
            edge_index = data.edge_index
            optimized_positions, mean, log_var = model(data.x, edge_index)
            dod_loss = domain_loss_fn(optimized_positions)
            mc_loss = minimal_loss_fn(original_positions, optimized_positions)
            diff_loss = diffusion_loss_fn(optimized_positions)
            print(f"Test Losses - Domain: {dod_loss:.4f}, Minimal Change: {mc_loss:.4f}, Diffusion: {diff_loss:.4f}")
            visualize_swarm(original_positions, title="Test: Original Swarm")
            visualize_swarm(optimized_positions, title="Test: Optimized Swarm")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_graphs = 510
    max_distance = 50
    grid_size = 1000
    input_dim = 5
    hidden_dim = 16
    output_dim = 2
    epochs = 25
    batch_size = 1
    taus = [1.0, 5.0, 10.0]
    r = 5.0
    min_distance = 5.0

    graphs = []
    for i in range(num_graphs):
        num_nodes = np.random.randint(30, 60) if i < 500 else 50
        positions = generate_agent_positions(num_nodes)
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        W = (distances < max_distance).astype(float)
        np.fill_diagonal(W, 0)
        edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
        x = torch.tensor(positions, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        data.original_positions = torch.tensor(positions, dtype=torch.float)
        data = add_graph_features(data, max_distance)
        graphs.append(data)

    train_loader = DataLoader(graphs[:500], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(graphs[500:], batch_size=batch_size, shuffle=False)

    model = GenerativeGNN(input_dim, output_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    domain_loss_fn = DomainOfDangerLoss(max_distance, grid_size)
    minimal_loss_fn = MinimalChangeLoss()
    diffusion_loss_fn = DiffusionLoss(taus, max_distance)
    kld_loss_fn = KLD()
    repulsion_loss_fn = RepulsionLoss(min_distance=min_distance)

    model = train_gnn(model, train_loader, optimizer, domain_loss_fn, minimal_loss_fn,
                      diffusion_loss_fn, kld_loss_fn, repulsion_loss_fn, epochs, device,
                      alpha=0.001, beta=15, gamma=2, delta=1, epsilon=1, tau=0.9)
    test_gnn(model, test_loader, device, domain_loss_fn, minimal_loss_fn, diffusion_loss_fn)
    return model

if __name__ == '__main__':
    main()

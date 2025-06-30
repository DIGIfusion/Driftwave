from PINN import PINN
from dataset import DriftwaveDataset
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.constants import pi

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    num_batches = len(dataloader)
    total_loss = 0.0
    for (X,y) in dataloader:
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred,y.unsqueeze(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        total_loss = total_loss + loss

    total_loss = total_loss / num_batches
    return total_loss


if __name__ == '__main__': 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = DriftwaveDataset()

    network = PINN()
    network.to(device)

    train_dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=7)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for epoch in range(50):
        print(f"Epoch: {epoch+1}")
        print("-----------------")
        loss = train_loop(train_dataloader, network, loss_fn, optimizer, 10, device)
        print(f"Loss: {loss}")

    torch.save(network.state_dict(), 'weights.pt')


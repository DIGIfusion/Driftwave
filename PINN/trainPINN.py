from PINN import PINN
from dataset import DriftwaveDataset
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.constants import pi

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device, v):
    num_batches = len(dataloader)
    total_loss_data = total_loss_pinn = 0.0
    for (X,y) in dataloader:
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss_data = loss_fn(pred,y.unsqueeze(-1))

        derivatives = torch.autograd.functional.jacobian(model,X)
        derivatives = torch.sum(derivatives,2)
        dndy = derivatives[:,:,0]
        dndt = derivatives[:,:,1]
        loss_pinn = loss_fn(-v*dndy,dndt)

        loss = loss_data + loss_pinn
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss_data = total_loss_data + loss_data
        total_loss_pinn = total_loss_pinn + loss_pinn

    total_loss_data = total_loss_data / num_batches
    total_loss_pinn = total_loss_pinn / num_batches
    return total_loss_data, total_loss_pinn

if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = DriftwaveDataset()

    network = PINN()
    network.to(device)

    T = 100
    Ln = 0.01
    v = - 2*pi*T/Ln

    train_dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=7)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for epoch in range(50):
        print(f"Epoch: {epoch+1}")
        print("-----------------")
        loss_data, loss_pinn = train_loop(train_dataloader, network, loss_fn, optimizer, 10, device, v)
        print(f"Data loss: {loss_data}, PINN loss: {loss_pinn}")

    torch.save(network.state_dict(), 'weights.pt')


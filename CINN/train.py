from CINN import CINN
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.constants import pi

class InitialDataset(Dataset):
    def __init__(self):
        self.resolution = 200
        self.L_y = 1.0
        self.ky = 1.0
        self.n0 = 1e19
        self.kygrid = torch.linspace(0, 2*pi*self.L_y, self.resolution)

    def __len__(self):
        return len(self.kygrid)

    def __getitem__(self,index):
        ntilde = torch.sin(self.ky*self.kygrid[index])
        t = torch.tensor(0.0)

        return torch.stack((self.kygrid[index],t),-1), ntilde

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    t_res = 1000

    num_batches = len(dataloader)
    total_initial_loss = 0.0
    total_boundary_loss = 0.0

    zero_grid = torch.zeros(t_res)
    twopi_grid = torch.full((t_res,),2*pi)
    t_grid = torch.linspace(0,1e-3,t_res)
    zero_input = torch.stack((zero_grid,t_grid),axis=1)
    twopi_input = torch.stack((twopi_grid,t_grid),axis=1)
    
    for (X,y) in dataloader:
        pred = model(X)
        loss_initial = loss_fn(pred,y.unsqueeze(-1))

        zero_output = model(zero_input)
        twopi_output = model(twopi_input)
        loss_boundary = loss_fn(zero_output,twopi_output)

        loss = loss_initial + loss_boundary

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        total_initial_loss = total_initial_loss + loss_initial
        total_boundary_loss = total_boundary_loss + loss_boundary

    total_initial_loss = total_initial_loss / num_batches
    total_boundary_loss = total_boundary_loss / num_batches
    
    return total_initial_loss, total_boundary_loss

T = 100
Ln = 0.01
v = - 2*pi*T/Ln

network = CINN(v)

dataset = InitialDataset()

train_dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

file_out = open("learning_curve.dat","w")
file_out.write("#epoch initial_condition_loss boundary_condition_loss\n")

for epoch in range(1000):
    print(f"Epoch: {epoch+1}")
    print("-----------------")
    loss_initial, loss_boundary = train_loop(train_dataloader, network, loss_fn, optimizer, 100)
    print(f"Initial condition loss: {loss_initial}")
    print(f"Boundary condition loss: {loss_boundary}")
    file_out.write(str(epoch+1) + " " + str(loss_initial.item()) + " " + str(loss_boundary.item()) + "\n")

file_out.close()

torch.save(network.state_dict(), 'weights.pt')


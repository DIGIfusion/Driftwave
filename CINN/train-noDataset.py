from CINN import CINN
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.constants import pi

def train_loop(model, loss_fn, optimizer):
    resolution = 200
    kygrid = torch.linspace(0, 2*pi, resolution)
    ntilde = torch.sin(kygrid)
    t0_grid = torch.zeros(resolution)
    initial_input = torch.stack((kygrid,t0_grid),axis=1)

    zero_grid = torch.zeros(1000)
    twopi_grid = torch.full((1000,),2*pi)
    t_grid = torch.linspace(0,1e-3,1000)
    zero_input = torch.stack((zero_grid,t_grid),axis=1)
    twopi_input = torch.stack((twopi_grid,t_grid),axis=1)
    
    pred = model(initial_input)
    loss_initial = loss_fn(pred,ntilde.unsqueeze(-1))

    zero_output = model(zero_input)
    twopi_output = model(twopi_input)
    loss_boundary = loss_fn(zero_output,twopi_output)

    loss = loss_initial + loss_boundary

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss_initial, loss_boundary

T = 100
Ln = 0.01
v = - 2*pi*T/Ln

network = CINN(v)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

for epoch in range(20000):
    print(f"Epoch: {epoch+1}")
    print("-----------------")
    loss_initial, loss_boundary = train_loop(network, loss_fn, optimizer)
    print(f"Initial condition loss: {loss_initial}")
    print(f"Boundary condition loss: {loss_boundary}")

torch.save(network.state_dict(), 'weights.pt')

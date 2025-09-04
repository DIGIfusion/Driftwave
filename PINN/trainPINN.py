from PINN import PINN
import torch
from torch import nn
from scipy.constants import pi
import random

def train_loop(model, loss_fn, optimizer, device, v):
    total_loss_pinn = 0.0

    y_grid = torch.linspace(0,2*pi,200, requires_grad=True)
    t_grid = torch.linspace(0,1e-3,100, requires_grad=True)

    ntilde = torch.sin(y_grid)
    ntilde = ntilde.to(device)
     
    initial_input = torch.stack((y_grid,torch.zeros(200)),dim=-1)
    initial_input = initial_input.to(device)

    initial_output = model(initial_input)

    loss_initial = loss_fn(initial_output.squeeze(),ntilde)

    zero_input = torch.stack((torch.zeros(100),t_grid),dim=-1)
    zero_input = zero_input.to(device)
    twopi_input = torch.stack((torch.full((100,),2*pi),t_grid),dim=-1)
    twopi_input = twopi_input.to(device)

    zero_output = model(zero_input)
    twopi_output = model(twopi_input)

    loss_boundary = loss_fn(zero_output,twopi_output)

    for i in range(1000):
        X = torch.stack((y_grid[random.randint(0,199)],t_grid[random.randint(0,99)]))
        X = X.to(device)

        derivatives = torch.autograd.functional.jacobian(model,X.unsqueeze(0),create_graph=True)
        derivatives = torch.sum(derivatives,2)
        dndy = derivatives[:,:,0]
        dndt = derivatives[:,:,1]
        loss_pinn = loss_fn(-v*dndy,dndt)

        total_loss_pinn = total_loss_pinn + loss_pinn

    total_loss_pinn = total_loss_pinn / 1000

    loss = loss_initial + loss_boundary + total_loss_pinn

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss_initial, loss_boundary, total_loss_pinn

if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network = PINN()
    network.to(device)

    T = 100
    Ln = 0.9
    v = - 2*pi*T/Ln

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    for epoch in range(1000):
        print(f"Epoch: {epoch+1}")
        print("-----------------")
        loss_initial, loss_boundary, loss_pinn = train_loop(network, loss_fn, optimizer, device, v)
        print(f"Initial loss: {loss_initial}, boundary loss: {loss_boundary}, PINN loss: {loss_pinn}")

    torch.save(network.state_dict(), 'weightsPINN.pt')


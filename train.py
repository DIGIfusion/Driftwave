import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from scipy.constants import pi
from NN import NN
from dataset import DriftwaveDataset 

resolution = 100
L_y = 1.0
kygrid = torch.linspace(0, 2*pi*L_y, resolution)
dky = torch.diff(kygrid)[0]/(2*pi*L_y)

def dndt(X, pred):
    dndt = torch.zeros(len(pred),len(pred[0]))

    for sample in pred:
        for i in range(len(pred[0])):
            dndt[:,i] = dndt[:,i] + torch.autograd.grad(sample[i],X,retain_graph=True)[0][:,2] # For each sample in batch

    dndt[:,-1]=dndt[:,0] # Periodic boundary condition
    return dndt

def comp_phi_dy(phi, order=4):
    dphidy = torch.zeros(len(phi),resolution)
    if order==2:
        dphidy[:,1:-1] = (phi[:,2:] - phi[:,0:-2])/(2*dky)
        dphidy[:,0] = (phi[:,1] - phi[:,-2])/(2*dky)
        dphidy[:,-1] = dphidy[:,0]
    if order==4:
        dphidy[:,2:-3] = (phi[:,0:-5] - 8*phi[:,1:-4] + 8*phi[:,3:-2] - phi[:,4:-1])/(12*dky)
        dphidy[:,-3] = (phi[:,-5] - 8*phi[:,-4] + 8*phi[:,-2] - phi[:,0])/(12*dky)
        dphidy[:,0] = (phi[:,-3] - 8*phi[:,-2] + 8*phi[:,1] - phi[:,2])/(12*dky)
        dphidy[:,1] = (phi[:,-2] - 8*phi[:,0] + 8*phi[:,2] - phi[:,3])/(12*dky)
        dphidy[:,-2] = (phi[:,-4] - 8*phi[:,-3] + 8*phi[:,0] - phi[:,1])/(12*dky)
        dphidy[:,-1] = dphidy[:,0]
    return dphidy

def comp_phi(delta_n, T=100):
    phi = T*delta_n
    return phi

def drift_X_grad(X, pred):
    phi = comp_phi(pred)
    dphidy = comp_phi_dy(phi) #electric field
    drift_X_grad = (1./X[:,1]).unsqueeze(1)*dphidy #multiply by gradient
    drift_X_grad[:,-1] = drift_X_grad[:,0] #periodic boundary condition
    return drift_X_grad

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss_data = loss_fn(pred,y)

        dn_dt = dndt(X, pred)
        drift_term = drift_X_grad(X,pred)
        loss_constraint = loss_fn(dn_dt,drift_term)

        loss = loss_data + loss_constraint

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), batch * batch_size + len(X)
        total_loss = total_loss + loss
        print(f"Batch train loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    total_loss = total_loss/num_batches
    return total_loss

def val_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    val_loss = 0.0
    with torch.no_grad():
        for (X, y) in dataloader:
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
    val_loss /= num_batches
    return val_loss

if __name__=='__main__':

    learning_rate = 2e-4
    N_epochs = 1000
    batch_size = 100

    model = NN()

    dataset = DriftwaveDataset()

    training, val = random_split(dataset, [0.8,0.2])

    train_dataloader = DataLoader(dataset = training, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(dataset = val, batch_size = batch_size, shuffle = False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(N_epochs):
        print(f"Epoch {epoch+1}\n---------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size)
        print(f"Total train loss: {train_loss}")
        val_loss = val_loop(val_dataloader, model, loss_fn)
        print(f"Total validation loss: {val_loss}")

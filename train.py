import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from scipy.constants import pi
from NN import NN
from dataset import DriftwaveDataset 
import copy
import os

resolution = 100
L_y = 1.0
kygrid = torch.linspace(0, 2*pi*L_y, resolution)
dky = torch.diff(kygrid)[0]/(2*pi*L_y)

def dndt(X, pred, device):
    dndt = torch.zeros(len(pred),len(pred[0]))
    dndt = dndt.to(device)

    for sample in pred:
        for i in range(len(pred[0])):
            dndt[:,i] = dndt[:,i] + torch.autograd.grad(sample[i],X,retain_graph=True)[0][:,2] # For each sample in batch

    dndt[:,-1]=dndt[:,0] # Periodic boundary condition
    dndt = 2e3 * dndt
    return dndt

def dndt_jacobian(X, model):
    jacobian = torch.autograd.functional.jacobian(model,X)
    dndt = torch.sum(jacobian,2)[:,:,2] #sum over derivates w.r.t. all batches and extract time derivate
    
    dndt[:,-1]=dndt[:,0] #periodic boundary condition
    dndt = 2e3 * dndt #rescale to real time
    return dndt

def comp_phi_dy(phi, device, order=4):
    dphidy = torch.zeros(len(phi),resolution)
    dphidy = dphidy.to(device)
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

def drift_X_grad(X, pred, device):
    phi = comp_phi(pred)
    dphidy = comp_phi_dy(phi, device) #electric field
    drift_X_grad = (1./X[:,1]).unsqueeze(1)*dphidy #multiply by gradient
    drift_X_grad[:,-1] = drift_X_grad[:,0] #periodic boundary condition
    return drift_X_grad

def train_loop(dataloader, model, loss_fn, optimizer, batch_size, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        X.requires_grad = True

        pred = model(X)
        loss_data = loss_fn(pred,y)

        #dn_dt = dndt(X, pred, device)                        #time derivative using grad
        dn_dt = dndt_jacobian(X, model)                       #time derivative using jacobian
        drift_term = drift_X_grad(X,pred,device)             #drift term in continuity equation
        loss_constraint = loss_fn(dn_dt,drift_term)

        loss = loss_data + 1e-10 * loss_constraint
        #loss = loss_data

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), batch * batch_size + len(X)
        total_loss = total_loss + loss
        print(f"Batch train loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    total_loss = total_loss/num_batches
    return total_loss

def val_loop(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    val_loss = 0.0
    with torch.no_grad():
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
    val_loss /= num_batches
    return val_loss

def predict(dataset, model, path, device):
    if not os.path.exists(path):
        os.makedirs(path)

    with torch.no_grad():
        for (X, y) in dataset:
            X = X.to(device)
            y = y.to(device)

            Ky = X[0].item()
            Ln = X[1].item()
            Ts = X[2].item()

            pred = model(X)

            with open(path + "/Ky" + str(Ky) + "_Ln" + str(Ln) + "_Ts" + str(Ts) + ".dat", "w") as f:
                for i, term in enumerate(y):
                    f.write(str(i) + " " + str(term.item()) + " " + str(pred[i].item()) + "\n")


if __name__=='__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate = 2e-4
    N_epochs = 10000
    batch_size = 100
    patience = 100

    pred_path = "predictions"

    model = NN()
    model.to(device)

    dataset = DriftwaveDataset()

    training, val = random_split(dataset, [0.8,0.2])

    train_dataloader = DataLoader(dataset = training, batch_size = batch_size, shuffle = True, num_workers=7)
    val_dataloader = DataLoader(dataset = val, batch_size = batch_size, shuffle = False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    best_val_loss = float('inf')
    best_model_weights = None

    loss_curve = open("loss_curve.dat","w")
    loss_curve.write("#epoch training_loss validation_loss\n")

    for epoch in range(N_epochs):
        print(f"Epoch {epoch+1}\n---------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, batch_size, device)
        print(f"Total train loss: {train_loss}")
        val_loss = val_loop(val_dataloader, model, loss_fn, device)
        print(f"Total validation loss: {val_loss}")

        loss_curve.write(str(epoch+1) + " " + str(train_loss) + " " + str(val_loss) + "\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_curr = patience
        else:
            patience_curr -= 1
            if patience_curr == 0:
                break

    loss_curve.close()

    model.load_state_dict(best_model_weights)

    torch.save(model.state_dict(), 'weights.pt')

    predict(val, model, pred_path, device)

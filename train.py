import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from NN import NN
from dataset import DriftwaveDataset 

def dndt(X, pred):
    dndt = torch.zeros(len(pred),len(pred[0]))

    for sample in pred:
        dndt[:,0] = dndt[:,0] + torch.autograd.grad(sample[0],X,retain_graph=True)[0][:,0]
        
    return dndt

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

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
    batch_size = 500

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

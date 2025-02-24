import torch
from torch import nn
from torch.utils.data import DataLoader
from NN import NN
from dataset import DriftwaveDataset 

def dndt(X, pred):
    dndt = torch.zeros(len(pred),len(pred[0]))

    for sample in pred:
        dndt[:,0] = dndt[:,0] + torch.autograd.grad(sample[0],X,retain_graph=True)[0][:,0]
        
    return dndt

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    num_batches = len(dataloader)
    total_loss = 0.0
    for (X, y) in dataloader:
        pred = model(X)

        #print(pred)
        #print(dndt(X, pred))

        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        loss = loss.item()
        total_loss = total_loss + loss

    total_loss = total_loss/num_batches
    return total_loss

if __name__=='__main__':

    learning_rate = 2e-4
    N_epochs = 1000
    batch_size = 4

    model = NN()

    dataset = DriftwaveDataset()

    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(N_epochs):
        print(f"Epoch {epoch+1}\n---------------------")
        train_loss = train_loop(dataloader, model, loss_fn, optimizer, batch_size)
        print(train_loss)

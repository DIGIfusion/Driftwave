#!/usr/bin/env python
# coding: utf-8

# In[1]:


from CINN import CINN
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.constants import pi


# In[2]:


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


# In[3]:


class BoundaryDataset(Dataset):
    def __init__(self):
        self.resolution = 2000
        self.tgrid = torch.linspace(0,1e-3,self.resolution)

    def __len__(self):
        return len(self.tgrid)

    def __getitem__(self,index):
        return torch.stack((torch.tensor(0.0),self.tgrid[index]),-1), torch.stack((torch.tensor(2*pi),self.tgrid[index]),-1)


# In[4]:


def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    num_batches = len(dataloader)
    total_loss = 0.0
    for (X,y) in dataloader:
        pred = model(X)
        loss = loss_fn(pred,y.unsqueeze(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        total_loss = total_loss + loss

    total_loss = total_loss / num_batches
    return total_loss


# In[5]:


def boundary_train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    num_batches = len(dataloader)
    total_loss = 0.0
    for (x0,xf) in dataloader:
        pred0 = model(x0)
        predf = model(xf)
        loss = loss_fn(pred0,predf)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        total_loss = total_loss + loss

    total_loss = total_loss / num_batches
    return total_loss


# In[6]:


T = 100
Ln = 0.01
v = - 2*pi*T/Ln


# In[7]:


network = CINN(v)


# In[8]:


dataset = InitialDataset()


# In[9]:


boundary_dataset = BoundaryDataset()


# In[10]:


train_dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)


# In[11]:


boundary_dataloader = DataLoader(dataset=boundary_dataset, batch_size=10, shuffle=True)


# In[12]:


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=2e-4)


# In[ ]:


for epoch in range(20000):
    print(f"Epoch: {epoch+1}")
    print("-----------------")
    loss_initial = train_loop(train_dataloader, network, loss_fn, optimizer, 10)
    print(f"Initial condition loss: {loss_initial}")

    loss_boundary = boundary_train_loop(boundary_dataloader, network, loss_fn, optimizer, 10)
    print(f"Boundary condition loss: {loss_boundary}")
    
    #loss_boundary = loss_fn(network(torch.tensor([0.0,1e-4])),network(torch.tensor([2*pi,1e-4])))
    #loss_boundary.backward()
    #optimizer.step()
    #optimizer.zero_grad()


# In[ ]:


torch.save(network.state_dict(), 'weights_ky=1_lr=2e-4_res_t=2000.pt')


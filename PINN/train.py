#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PINN import PINN
from dataset import DriftwaveDataset
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.constants import pi


# In[2]:


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


# In[3]:


dataset = DriftwaveDataset()


# In[4]:


network = PINN()


# In[5]:


train_dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)


# In[6]:


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)


# In[7]:


for epoch in range(50):
    print(f"Epoch: {epoch+1}")
    print("-----------------")
    loss = train_loop(train_dataloader, network, loss_fn, optimizer, 10)
    print(f"Loss: {loss}")


# In[ ]:


torch.save(network.state_dict(), 'weights.pt')


import torch
from torch import nn

class CINN(nn.Module):
    def __init__(self, v):
        super(CINN, self).__init__()

        self.transform = nn.Linear(2,1,bias=False)    #characteristic transform
        self.transform.weight.requires_grad = False
        self.transform.weight[0,0] = 1.0
        self.transform.weight[0,1] = -v

        self.linear1 = nn.Linear(1,50)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(50,50)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(50,50)
        self.tanh3 = nn.Tanh()
        self.linear4 = nn.Linear(50,1)
        self.tanh4 = nn.Tanh()

    def forward(self, x):
        x = self.transform(x)
        
        x = self.linear1(x)
        x = self.tanh1(x)
        x = self.linear2(x)
        x = self.tanh2(x)
        x = self.linear3(x)
        x = self.tanh3(x)
        x = self.linear4(x)

        x = self.tanh4(x)
        
        return x
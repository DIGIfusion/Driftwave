import torch
from torch import nn

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.linear1 = nn.Linear(3,1000)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1000,1000)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1000,100)

        self.tanh3 = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        x = self.tanh3(x)

        return x

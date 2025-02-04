import torch
from torch import nn

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.linear1 = nn.Linear(3,20)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(20,20)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(20,20)
        self.relu3 = nn.ReLU()

        self.linear4 = nn.Linear(20,100)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)

        x = self.linear4(x)

        return x

import torch
from torch.utils.data import Dataset

class DriftwaveDataset(Dataset):
    def __init__(self):
        self.input = torch.rand(50,3,requires_grad=True)
        self.output = torch.rand(50,100)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.output[index]

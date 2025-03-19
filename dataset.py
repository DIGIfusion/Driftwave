import os
import torch
from torch.utils.data import Dataset
import pandas as pd

class DriftwaveDataset(Dataset):
    def __init__(self):
        self.list = [val for sublist in [[os.path.join(root,name) for name in file] for root, dir, file in os.walk("./NewDataset")] for val in sublist]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        df = pd.read_csv(self.list[index])

        Ky = df.iloc[0].loc['Ky']
        Ln = df.iloc[0].loc['Ln']
        Ts = 2.0*df.iloc[0].loc['Ts']/1e-3 - 1.0

        sol = torch.tensor(df['sol']/1e17)

        return torch.tensor([Ky,Ln,Ts]).float(), sol.float()

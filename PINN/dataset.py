import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.constants import pi

class DriftwaveDataset(Dataset):
    def __init__(self):
        self.list = [val for sublist in [[os.path.join(root,name) for name in file] for root, dir, file in os.walk("../NewDataset/Ln0.01Ky1")] for val in sublist]
        self.resolution = 100
        self.kygrid = torch.linspace(0,2*pi,self.resolution)

    def __len__(self):
        return len(self.list)*self.resolution

    def __getitem__(self, index):
        df = pd.read_csv(self.list[int(index/self.resolution)])
        
        y = self.kygrid[index%self.resolution]
        Ts = df.iloc[0].loc['Ts']

        sol = torch.tensor(df['sol']/1e17)[index%self.resolution]

        return torch.tensor([y,Ts]).float(), sol.float()

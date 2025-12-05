import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset Definition
class Gen2DDataset(Dataset):
    def __init__(self,
                 mask = lambda x, y: True,
                 num_samples=5000,
                 fill_till_num_samples = False,
                 lims = np.array([1, 1]),
                 funs = lambda x, i: x):
                 
        self.data = []
        self.targets = []
        if fill_till_num_samples:
            while (len(self.data) < num_samples):
                xy = np.random.rand(2)  # Use continuous values for x and y
                xy = - lims + 2 * xy * lims
                if mask(*xy):
                    z = funs(funs(xy[0], 0) + funs(xy[1], 1), 2)
                    self.data.append(xy)  # Store x and y as a 2D vector
                    self.targets.append(np.array([z]))
        else:
            for i in range(num_samples):
                xy = np.random.rand(2)  # Use continuous values for x and y
                xy = - lims + 2 * xy * lims
                if mask(*xy):
                    z = funs(funs(xy[0], 0) + funs(xy[1], 1), 2)
                    self.data.append(xy)  # Store x and y as a 2D vector
                    self.targets.append(np.array([z]))

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
    
    def get_all_data(self):
        # Return all data and targets as tensors
        inputs = torch.tensor(self.data, dtype=torch.float32)
        outputs = torch.tensor(self.targets, dtype=torch.float32)
        return inputs, outputs
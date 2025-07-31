import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset Definition
class GeneralDataset(Dataset):
    def __init__(self,
                 encode_num = 4,
                 num_samples = 1000,
                 mean = 0,
                 std_dev = 2,
                 mapfun = lambda x : x):

        #######################################
        ### Draw from uniform distributions ###
        #######################################

        self.data = (np.random.rand(num_samples, encode_num) - 0.5) * 2 * std_dev + mean

        ########################
        ### Generate outputs ###
        ########################

        self.num_samples = len(self.data)
        self.mapfun = mapfun
        self.__generate__()

    def __generate__(self):
        self.targets = []
        for i in range(self.num_samples):
            self.targets.append(self.mapfun(self.data[i]))
        self.targets = np.array(self.targets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

    def get_all_data(self):
        # Return all data and targets as tensors
        inputs = torch.tensor(self.data, dtype=torch.float32)
        outputs = torch.tensor(self.targets, dtype=torch.float32)
        return inputs, outputs
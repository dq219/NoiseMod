import torch
import torch.nn as nn
import numpy as np

class MODELMixedModular(nn.Module):
    def __init__(self,
                 encode_num = 4, decode_num = 4, module_size = 16, hidden_num = 64, module_layer = 3,
                 hidden_layer = 1):
        super(MODELMixedModular, self).__init__()

        self.module_size = module_size

        modules = []
        for _1 in range(encode_num):
            mod = []
            mod.append(nn.Linear(1, self.module_size))
            mod.append(nn.LeakyReLU(negative_slope=0.1))
            for _2 in range(module_layer - 1):
                mod.append(nn.Linear(self.module_size, self.module_size))
                mod.append(nn.LeakyReLU(negative_slope=0.1))
            mod.append(nn.Linear(self.module_size, 1))
            mod.append(nn.LeakyReLU(negative_slope=0.1))
            mod = nn.Sequential(*mod)
            modules.append(mod)
        self.modlayer = nn.ModuleList(modules)

        layers = []
        if hidden_layer > 0:
            layers.append(nn.Linear(encode_num, hidden_num))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            for i in range(hidden_layer - 1):
                layers.append(nn.Linear(hidden_num, hidden_num))
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Linear(hidden_num, decode_num))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        if len(self.modlayer) > 0:
            xx = []
            for i, modlyr in enumerate(self.modlayer):
                xx.append(modlyr(x[:, i].unsqueeze(-1)))
            xx = torch.cat(xx, dim=1)
        else:
            xx = x
        for l in self.layer:
            xx = l(xx)
        return xx

    def fit(self, train_dataloader, train_dataset, test_dataset, optimizer, criterion, device = torch.device('cpu')):
        LOSS = [0, 0]

        self.train()
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        self.eval()
        inputs, targets = train_dataset.get_all_data()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self(inputs)
        LOSS[0] += criterion(outputs, targets).item()

        inputs, targets = test_dataset.get_all_data()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self(inputs)
        loss = criterion(outputs, targets)
        LOSS[1] += criterion(outputs, targets).item()

        return LOSS

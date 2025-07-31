import torch.nn as nn
import torch
import numpy as np


class modelModular(nn.Module):
    def __init__(self, L_single = 64, L_post = 128, N_post = 0):
        super(modelModular, self).__init__()
        
        # Define the MLPs for x and y
        self.mlp_02_x = nn.Sequential(
            nn.Linear(1, L_single),
            nn.PReLU(),
            nn.Linear(L_single, 1)
        )
        self.mlp_02_y = nn.Sequential(
            nn.Linear(1, L_single),
            nn.PReLU(),
            nn.Linear(L_single, 1)
        )
        self.mlp_23 = nn.Sequential(
            nn.PReLU(),
            nn.Linear(2, L_post),
            nn.PReLU(),
        )
        final_layers = []
        for i in range(N_post):
            final_layers.append(nn.Linear(L_post, L_post))
            final_layers.append(nn.PReLU())
        final_layers.append(nn.Linear(L_post, 1))
        self.mlp_35 = nn.Sequential(*final_layers)

    def forward(self, xy):
        x2 = self.mlp_02_x(xy[:, 0].unsqueeze(-1))
        y2 = self.mlp_02_y(xy[:, 1].unsqueeze(-1))
        combined = torch.cat((x2, y2), dim=1)
        r2 = self.mlp_23(combined)
        z = self.mlp_35(r2)
        return z

# full mlp definition, takes total parameter count and layer size into consideration
class modelBottleNeck(nn.Module):
    def __init__(self, L_hidden = 128, N_hidden = 0):
        super(modelBottleNeck, self).__init__()
        layers = []

        layers.append(nn.Linear(2, L_hidden))
        layers.append(nn.PReLU())
        layers.append(nn.Linear(L_hidden, 2))
        layers.append(nn.PReLU())
        layers.append(nn.Linear(2, L_hidden))
        layers.append(nn.PReLU())

        for i in range(N_hidden):
            layers.append(nn.Linear(L_hidden, L_hidden))
            layers.append(nn.PReLU())

        # Final layer: map to 1 channel with correct output size
        layers.append(nn.Linear(L_hidden, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x

# full mlp definition, takes total parameter count and layer size into consideration
class modelFull(nn.Module):
    def __init__(self, L_hidden = 128, N_hidden = 2):
        super(modelFull, self).__init__()
        layers = []

        layers.append(nn.Linear(2, L_hidden))
        layers.append(nn.PReLU())
        
        for i in range(N_hidden):
            layers.append(nn.Linear(L_hidden, L_hidden))
            layers.append(nn.PReLU())

        # Final layer: map to 1 channel with correct output size
        layers.append(nn.Linear(L_hidden, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
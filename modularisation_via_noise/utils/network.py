import torch
import torch.nn as nn
import numpy as np

def regularizer_Act(acts, alpha = 2):
    res = 0
    for i in range(len(acts)):
        act = acts[i]
        pass_error = (torch.abs(act) ** alpha)
        res += torch.mean(torch.sum(pass_error, axis = 1))
    return res

def regularizer_WeightedAct(layers, acts, alpha = 2):
    wt_idxes = []
    for idx in range(len(layers)):
        if isinstance(layers[idx], nn.Linear):
            wt_idxes.append(idx)
    res = 0
    for i in range(len(acts)):
        act = acts[i]
        pass_error = (torch.abs(act) ** alpha) @ (layers[wt_idxes[i + 1]].weight.T ** 2)
        res += torch.mean(torch.sum(pass_error, axis = 1))
    return res

def regularizer_L12(layers):
    wt_idxes = []
    for idx in range(len(layers)):
        if isinstance(layers[idx], nn.Linear):
            wt_idxes.append(idx)
    res = 0
    for i in range(len(wt_idxes) - 1):
        enc_weights = layers[wt_idxes[i]].weight
        dec_weights = layers[wt_idxes[i + 1]].weight
        (enc_out, enc_in) = enc_weights.shape
        (dec_out, dec_in) = dec_weights.shape
        encoder_norms = torch.sum(torch.abs(enc_weights), dim = 1)
        decoder_norms = torch.sum(dec_weights ** 2, dim = 0)
        if enc_out == dec_in:
            res += torch.sum(encoder_norms * decoder_norms)
    return res

def regularizer_L1(layers):
    res = 0
    for idx in range(len(layers)):
        if isinstance(layers[idx], nn.Linear):
            weights = layers[idx].weight
            res += torch.sum(torch.abs(weights))
    return res

def regularizer_L2(layers):
    res = 0
    for idx in range(len(layers)):
        if isinstance(layers[idx], nn.Linear):
            weights = layers[idx].weight
            res += torch.sum(weights ** 2)
    return res

class NoiseInject(nn.Module):
    def __init__(self, coef = -1, expt = -1):
        super(NoiseInject, self).__init__()
        self.coef = coef
        self.expt = expt

    def forward(self, x):
        # if self.training and (self.coef > 0):
        if self.coef > 0:
            if self.expt > 0:
                noise = self.coef * torch.randn_like(x) * (torch.abs(x) ** (self.expt * 0.5))
            else:
                noise = self.coef * torch.randn_like(x)
            return x + noise
        else:
            return x

class MODEL(nn.Module):
    def __init__(self,
                 encode_num = 4, decode_num = 4, hidden_num = 64, bias = True, nonlinear = True, hidden_layer = 3, use_tanh = False, use_sigmoid = False, # network-specific parameters
                 L1 = -1, L2 = -1, L12 = -1, WeightedAct = -1, Act = -1, alpha = 2, dropout = -1, noise_coef = -1, noise_expt = 1, # noise/regularisers
                 **kwargs):

        super(MODEL, self).__init__()

        self.L1 = L1
        self.L2 = L2
        self.L12 = L12
        self.Act = Act
        self.WeightedAct = WeightedAct
        self.alpha = alpha

        layers = []
        layers.append(nn.Linear(encode_num, hidden_num, bias = bias))

        if nonlinear:
            if use_tanh:
                layers.append(nn.Tanh())
            elif use_sigmoid:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.1))
        if dropout > 0:
            layers.append(nn.Dropout(p = dropout))
        if noise_coef > 0:
            layers.append(NoiseInject(coef = noise_coef, expt = noise_expt))

        for i in range(hidden_layer - 1):
            layers.append(nn.Linear(hidden_num, hidden_num, bias = bias))
            if nonlinear:
                if use_tanh:
                    layers.append(nn.Tanh())
                elif use_sigmoid:
                    layers.append(nn.Sigmoid())
                else:
                    layers.append(nn.LeakyReLU(negative_slope=0.1))
            if dropout > 0:
                layers.append(nn.Dropout(p = dropout))
            if noise_coef > 0:
                layers.append(NoiseInject(coef = noise_coef, expt = noise_expt))

        layers.append(nn.Linear(hidden_num, decode_num, bias = bias))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        activations = []
        # activations.append(x)
        for l in self.layer:
            x = l(x)
            if (isinstance(l, nn.LeakyReLU) | isinstance(l, nn.Tanh) | isinstance(l, nn.Sigmoid)):
                activations.append(x)
        return x, activations

    def fit(self, train_dataloader, train_dataset, test_dataset, optimizer, criterion, device = torch.device('cpu')):
        # losses are arranged in the following order:
        # train, test, L1, L2, LWA, LA, and L12 losses
        LOSS = [0, 0, 0, 0, 0, 0, 0] 
        self.train()
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, acts = self(inputs)
            loss = criterion(outputs, targets)

            if self.L1 > 0:
                loss_regu_L1 = regularizer_L1(self.layer)
                LOSS[2] += loss_regu_L1.item() / len(train_dataloader)
                loss += loss_regu_L1 * self.L1
            if self.L2 > 0:
                loss_regu_L2 = regularizer_L2(self.layer)
                LOSS[3] += loss_regu_L2.item() / len(train_dataloader)
                loss += loss_regu_L2 * self.L2
            if self.WeightedAct > 0:
                loss_regu_WA = regularizer_WeightedAct(self.layer, acts, self.alpha)
                LOSS[4] += loss_regu_WA.item() / len(train_dataloader)
                loss += loss_regu_WA * self.WeightedAct
            if self.Act > 0:
                loss_regu_AA = regularizer_Act(acts, self.alpha)
                LOSS[5] += loss_regu_AA.item() / len(train_dataloader)
                loss += loss_regu_AA * self.Act
            if self.L12 > 0:
                loss_regu_L12 = regularizer_L12(self.layer)
                LOSS[6] += loss_regu_L12.item() / len(train_dataloader)
                loss += loss_regu_L12 * self.L12

            loss.backward()
            optimizer.step()

        self.eval()
        inputs, targets = train_dataset.get_all_data()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, acts = self(inputs)
        LOSS[0] += criterion(outputs, targets).item()

        inputs, targets = test_dataset.get_all_data()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, acts = self(inputs)
        loss = criterion(outputs, targets)
        LOSS[1] += criterion(outputs, targets).item()

        return LOSS

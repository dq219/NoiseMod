from data import ConcatCIFAR10
import torch
from torch.utils.data import DataLoader
from training_utils import run_one

data_root = 'data'
D = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = ConcatCIFAR10(root=data_root, D=D, train=True)
test_set = ConcatCIFAR10(root=data_root, D=D, train=False)

train_loader = DataLoader(
    train_set, batch_size=32, shuffle=True,
    num_workers=4, pin_memory=(device.type == "cuda"),
    drop_last=True)
test_loader = DataLoader(
    test_set, batch_size=256, shuffle=False,
    num_workers=4, pin_memory=(device.type == "cuda"))

reg_weights = [0.0, 1.0]
for reg_weight in reg_weights:
    run_one(reg_weight, train_loader, test_loader, device, epochs = 500)
import torch
from torch import nn

def save_weights(model: nn.Module, filename: str ="weights.pth"):
    torch.save(model.state_dict(), filename)

def load_weights(model: nn.Module, filename: str ="weights.pth"):
    model.load_state_dict(torch.load(filename))
    return model



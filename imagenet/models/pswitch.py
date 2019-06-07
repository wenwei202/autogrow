import torch
import torch.nn as nn

class PSwitch(nn.Module):
    """
     This is a learnable switch
    """
    def __init__(self, value=1.0):
        super(PSwitch, self).__init__()
        self.switch = nn.Parameter(torch.Tensor(1))
        self.switch.data.fill_(value)

    def forward(self, input):
        return input * self.switch

    def get_switch(self):
        return self.switch.data
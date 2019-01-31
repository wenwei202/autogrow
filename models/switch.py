import torch
import torch.nn as nn

class Switch(nn.Module):
    """
     This is a param-free switch
    """
    def __init__(self, value=1.0, steps=1, start=0.0, stop=1.0, mode='linear'):
        super(Switch, self).__init__()
        self.value = torch.ones(()) * value
        self.steps = steps
        self.start = start
        self.stop = stop
        self.mode = mode
        assert (self.steps >= 1)
        assert (self.stop >= self.start)
        self.register_buffer('switch', self.value)

    def set_params(self, steps, start=0.0, stop=1.0, mode='linear'):
        self.steps = steps
        self.start = start
        self.stop = stop
        self.mode = mode
        assert (self.steps >= 1)
        assert (self.stop >= self.start)

    def forward(self, input):
        return input * self.switch

    def increase(self):
        if 'linear' == self.mode:
            self.switch += (self.stop - self.start) / self.steps

        self.switch = self.start if self.switch < self.start else self.switch
        self.switch = self.stop if self.switch > self.stop else self.switch

    def get_switch(self):
        return self.switch

    def extra_repr(self):
        switch_str = '%.3f' % self.switch
        return switch_str
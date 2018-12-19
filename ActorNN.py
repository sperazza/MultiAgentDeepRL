import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, pred_state_size, state_size, action_size, seed, fc1_units=600, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(pred_state_size + state_size, fc1_units)
        self.do1 = nn.Dropout(.2)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc5 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, prev_state, state):
        """Build an actor (policy) network that maps states -> actions."""

        pred_state = state + (state - prev_state)
        xs = torch.cat((pred_state, state), dim=1)

        x = self.fc1(xs)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.do1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.tanh(x)

        x = self.fc5(x)
        x = F.tanh(x)

        return x

import torch
import torch.nn as nn

class BCActorNetwork(nn.Module):

    def __init__(self, state_dims,action_dims):
        super().__init__()

        self.network=nn.Sequential(
            nn.Linear(state_dims,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_dims),
            nn.Tanh()
        )

    def forward(self,x):

        return self.network(x)
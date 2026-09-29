import torch
import torch.nn as nn


# ==========================
# PPO Expert Actor
# ==========================

class PPOActorNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(state_dim,64),
            nn.ReLU(),

            nn.Linear(64,64),
            nn.ReLU(),

            nn.Linear(64,action_dim)

        )


        # PPO中的可学习标准差

        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )


    def forward(self,x):

        mean = self.network(x)

        std = torch.exp(
            self.log_std
        )

        return mean,std



# ==========================
# BC / DAgger Learner Actor
# ==========================

class BCActorNetwork(nn.Module):

    def __init__(self,state_dim,action_dim):

        super().__init__()


        self.network = nn.Sequential(

            nn.Linear(state_dim,64),
            nn.ReLU(),

            nn.Linear(64,64),
            nn.ReLU(),

            nn.Linear(64,action_dim),

            nn.Tanh()

        )


    def forward(self,x):

        action = self.network(x)

        return action
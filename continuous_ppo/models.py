import torch 
import torch.nn as nn

class ActorNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        # 主网络：
        # state_dim
        # -> 64
        # -> ReLU
        # -> 64
        # -> ReLU
        # -> action_dim
        #
        # 输出 mean

        self.network = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_dim)
        )


        # 可学习的 log_std
        #
        # shape = [action_dim]
        #
        # 初始值先全部设为0

        self.log_std = nn.Parameter(torch.zeros(action_dim))


    def forward(self, x):

        # 1. 计算 mean
        mean = self.network(x)

        # 2. log_std -> std
        std = torch.exp(self.log_std)

        # 3. 返回 mean 和 std
        return mean,std

class CriticNetwork(nn.Module):

    def __init__(self,state_dim):
        super().__init__()

        self.network=nn.Sequential(

            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)

        )

    def forward(self,x):

        value = self.network(x)

        return value.squeeze(-1)
# import numpy as np
# import torch
# import torch.nn as nn

# from environment import GridWorld

# class QNetwork(nn.Module):

#     def __init__(
#         self,
#         state_dim,
#         action_dim
#     ):

#         super().__init__()

#         self.network = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim)
#         )


#     def forward(self, x):

#         return self.network(x)

# def state_to_tensor(
#     state,
#     rows,
#     cols
# ):

#     row, col = state

#     state_index = (
#         row * cols + col
#     )

#     one_hot = np.zeros(
#         rows * cols,
#         dtype=np.float32
#     )

#     one_hot[state_index] = 1.0

#     return torch.tensor(
#         one_hot,
#         dtype=torch.float32
#     )

# env = GridWorld()

# state_dim = (
#     env.rows * env.cols
# )

# action_dim = 4


# q_network = QNetwork(
#     state_dim,
#     action_dim
# )

# state = (2, 2)

# state_tensor = state_to_tensor(
#     state,
#     env.rows,
#     env.cols
# )

# print(
#     "State:",
#     state
# )

# print(
#     "State tensor:",
#     state_tensor
# )

# print(
#     "Shape:",
#     state_tensor.shape
# )

# q_values = q_network(
#     state_tensor
# )

# print(
#     "\nQ values:"
# )

# print(
#     q_values
# )

# print(
#     "Q shape:",
#     q_values.shape
# )

# import numpy as np

# import torch
# import torch.nn as nn

# from environment import GridWorld


# # =========================
# # 1. Random Seed
# # =========================

# seed = 42

# rng = np.random.default_rng(seed)

# torch.manual_seed(seed)


# # =========================
# # 2. Environment
# # =========================

# env = GridWorld()

# state_dim = 16 #把4x4 转换为 1x16存在数组里面输入给神经网络

# action_dim = 4


# # =========================
# # 3. Q Network
# # =========================

# class QNetwork(nn.Module):

#     def __init__(
#         self,
#         state_dim,
#         action_dim
#     ):

#         super().__init__()

#         # TODO:
#         #
#         # 建立一个网络：
#         #
#         # state_dim
#         #     ↓
#         # 64
#         #     ↓
#         # action_dim
#         #
#         # 中间使用 ReLU

#         self.network = nn.Sequential(
#             nn.Linear(state_dim,64),
#             nn.ReLU(),
#             nn.Linear(64,action_dim)
#         )


#     def forward(self, x):

#         # TODO:
#         # 输入 state tensor
#         # 输出每个动作的 Q 值

#         return self.network(x)


# # =========================
# # 4. State -> Tensor
# # =========================

# def state_to_tensor(
#     state,
#     rows,
#     cols
# ):

#     row, col = state

#     # TODO:
#     #
#     # 把二维位置：
#     # (row, col)
#     #
#     # 转成 0~15 的 state index
#     #
#     # 例如：
#     # (0,0) -> 0
#     # (0,1) -> 1
#     # (1,0) -> 4
#     # (2,2) -> 10

#     state_index = row * cols + col

#     # TODO:
#     # 创建长度 rows * cols 的全 0 数组

    
#     one_hot = np.zeros(
#         rows * cols,
#         dtype = np.float32
#     )


#     # TODO:
#     # 对应 state_index 的位置设为 1

#     one_hot[state_index] = 1


#     # TODO:
#     # 转换成 float32 torch tensor
#     #tensor 是专门服务于神经网络的多维数组!!!!
#     state_tensor = torch.tensor(
#     one_hot,
#     dtype=torch.float32
# )


#     return state_tensor


# # =========================
# # 5. Create Network
# # =========================

# q_network = QNetwork(
#     state_dim,
#     action_dim
# )


# # =========================
# # 6. Test One State
# # =========================

# state = (2, 2)

# state_tensor = state_to_tensor(
#     state,
#     env.rows,
#     env.cols
# )


# print(
#     "State:"
# )

# print(
#     state
# )


# print(
#     "\nState Tensor:"
# )

# print(
#     state_tensor
# )


# print(
#     "\nState Tensor Shape:"
# )

# print(
#     state_tensor.shape
# )


# # =========================
# # 7. Forward
# # =========================

# q_values = q_network(
#     state_tensor
# )


# print(
#     "\nQ Values:"
# )

# print(
#     q_values
# )


# print(
#     "\nQ Values Shape:"
# )

# print(
#     q_values.shape
# )


# # =========================
# # 8. Greedy Action
# # =========================

# # TODO:
# #
# # 从四个 Q 值中找出最大值对应的 action
# #
# # 最终 action 必须是 Python / integer scalar
# # 例如：
# #
# # action = 2

# action = torch.argmax(q_values).item() #改为用pytorch的写法


# print(
#     "\nGreedy Action:"
# )

# print(
#     action
# )


import numpy as np 
import torch 
import torch.nn as nn
from environment import GridWorld

env = GridWorld()
rng = np.random.default_rng(42)
state_dims = 16
action_dims = 4
torch.manual_seed(42)
#定义一个q网络的类，这个类初始化网路参数，每个层以及定义前向传播
class QNetwork(nn.Module):

    def __init__(self,
                 state_dims,
                 action_dims):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dims,64),
            nn.ReLU(),
            nn.Linear(64,action_dims)
                        )

    def forward(self,x):

        return self.network(x)

def state_to_tensor(state):

    row,col = state

    state_index = env.cols * row + col

    one_hot = np.zeros(env.rows * env.cols,dtype=np.float32)

    one_hot[state_index] = 1

    state_tensor = torch.tensor(one_hot,dtype=torch.float32)

    return state_tensor

q_network = QNetwork(state_dims,action_dims)

state = (2,2)

state_tensor = state_to_tensor(state)

print("state:",state)

print("\nstate_tensor:",state_tensor)

q_values = q_network(state_tensor)

print("\nq_values:",q_values)

action = torch.argmax(q_values).item()

print("\naction:",action)

def select_action(state, epsilon):

    # TODO 1:
    # 以 epsilon 概率随机探索
    if rng.random() < epsilon:

        action = rng.integers(4)

    else:

        # TODO 2:
        # state -> tensor
        state_tensor = state_to_tensor(state)


        # TODO 3:
        # 用 Q Network 计算四个 Q 值
        #
        # 注意：
        # 这里只是在选动作，不需要计算梯度
        with torch.no_grad():

            q_values = q_network(state_tensor)


        # TODO 4:
        # 找 Q 最大的动作
        action = torch.argmax(q_values).item()


    return action

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from collections import deque

# from environment import GridWorld


# # =========================
# # 1. Random Seed
# # =========================

# torch.manual_seed(42)


# # =========================
# # 2. Q Network
# # =========================

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


# # =========================
# # 3. Replay Buffer
# # =========================

# class ReplayBuffer:

#     def __init__(
#         self,
#         capacity
#     ):

#         self.buffer = deque(
#             maxlen=capacity
#         )


#     def push(
#         self,
#         state,
#         action,
#         reward,
#         next_state,
#         done
#     ):

#         self.buffer.append(
#             (
#                 state,
#                 action,
#                 reward,
#                 next_state,
#                 done
#             )
#         )


#     def sample(
#         self,
#         batch_size,
#         rng
#     ):

#         indices = rng.choice(
#             len(self.buffer),
#             size=batch_size,
#             replace=False
#         )

#         batch = [
#             self.buffer[i]
#             for i in indices
#         ]

#         return batch


#     def __len__(self):

#         return len(
#             self.buffer
#         )


# # =========================
# # 4. State -> Tensor
# # =========================

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


# # =========================
# # 5. Environment
# # =========================

# env = GridWorld()

# rng = np.random.default_rng(
#     42
# )

# state_dim = (
#     env.rows * env.cols
# )

# action_dim = 4


# # =========================
# # 6. Q Network
# # =========================

# q_network = QNetwork(
#     state_dim,
#     action_dim
# )


# # =========================
# # 7. Optimizer + Loss
# # =========================

# optimizer = optim.Adam(
#     q_network.parameters(),
#     lr=0.01
# )

# loss_fn = nn.MSELoss()


# # =========================
# # 8. Hyperparameters
# # =========================

# gamma = 1.0

# batch_size = 8


# # =========================
# # 9. Replay Buffer
# # =========================

# buffer = ReplayBuffer(
#     capacity=500
# )


# # =========================
# # 10. Collect Experience
# # =========================

# state = env.reset()


# for step in range(200):

#     action = int(
#         rng.integers(4)
#     )

#     next_state, reward, done = env.step(
#         action
#     )


#     buffer.push(
#         state,
#         action,
#         reward,
#         next_state,
#         done
#     )


#     state = next_state


#     if done:

#         state = env.reset()


# print(
#     "Buffer size:",
#     len(buffer)
# )


# # =========================
# # 11. Sample Batch
# # =========================

# batch = buffer.sample(
#     batch_size,
#     rng
# )


# # =========================
# # 12. Separate Batch
# # =========================

# states = []
# actions = []
# rewards = []
# next_states = []
# dones = []


# for transition in batch:

#     state, action, reward, next_state, done = transition

#     states.append(state)
#     actions.append(action)
#     rewards.append(reward)
#     next_states.append(next_state)
#     dones.append(done)


# # =========================
# # 13. Convert To Tensors
# # =========================

# states_tensor = torch.stack(
#     [
#         state_to_tensor(
#             state,
#             env.rows,
#             env.cols
#         )
#         for state in states
#     ]
# )


# next_states_tensor = torch.stack(
#     [
#         state_to_tensor(
#             state,
#             env.rows,
#             env.cols
#         )
#         for state in next_states
#     ]
# )


# actions_tensor = torch.tensor(
#     actions,
#     dtype=torch.long
# ).unsqueeze(1)


# rewards_tensor = torch.tensor(
#     rewards,
#     dtype=torch.float32
# )


# dones_tensor = torch.tensor(
#     dones,
#     dtype=torch.float32
# )


# # =========================
# # 14. Q(s, all actions)
# # =========================

# q_values = q_network(
#     states_tensor
# )


# print(
#     "\nQ values shape:",
#     q_values.shape
# )

# print(
#     "Q values:"
# )

# print(
#     q_values.detach()
# )


# # =========================
# # 15. Select Q(s,a)
# # =========================

# predicted_q = q_values.gather(
#     1,
#     actions_tensor
# ).squeeze(1)


# print(
#     "\nActions:"
# )

# print(
#     actions_tensor.squeeze(1)
# )


# print(
#     "\nPredicted Q:"
# )

# print(
#     predicted_q.detach()
# )


# print(
#     "Predicted Q shape:",
#     predicted_q.shape
# )


# # =========================
# # 16. TD Targets
# # =========================

# with torch.no_grad():

#     next_q_values = q_network(
#         next_states_tensor
#     )

#     best_next_q = torch.max(
#         next_q_values,
#         dim=1
#     ).values


#     targets = (
#         rewards_tensor
#         +
#         gamma
#         *
#         best_next_q
#         *
#         (1.0 - dones_tensor)
#     )


# print(
#     "\nTargets:"
# )

# print(
#     targets
# )


# print(
#     "Targets shape:",
#     targets.shape
# )


# # =========================
# # 17. Batch Loss
# # =========================

# loss = loss_fn(
#     predicted_q,
#     targets
# )


# print(
#     "\nLoss:",
#     loss.item()
# )


# # =========================
# # 18. Gradient Update
# # =========================

# optimizer.zero_grad()

# loss.backward()

# optimizer.step()


# # =========================
# # 19. Predict Again
# # =========================

# new_q_values = q_network(
#     states_tensor
# )

# new_predicted_q = new_q_values.gather(
#     1,
#     actions_tensor
# ).squeeze(1)


# print(
#     "\nPredicted Q After Update:"
# )

# print(
#     new_predicted_q.detach()
# )


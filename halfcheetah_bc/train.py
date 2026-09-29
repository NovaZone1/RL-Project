import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from models import BCActorNetwork

data=np.load("expert_dataset.npz")
states=data["states"]
actions=data["actions"]

states_tensor=torch.tensor(states,dtype=torch.float32)
actions_tensor=torch.tensor(actions,dtype=torch.float32)

state_dims=17
action_dims=6

bc_actor_network=BCActorNetwork(state_dims,action_dims)
optimizer=optim.Adam(
    bc_actor_network.parameters(),
    lr=3e-4
)

loss_fun=nn.MSELoss()

epochs=100
batch_size=256
dataset_size=len(states)

for epoch in range(epochs):

    indices=torch.randperm(dataset_size)
    total_loss=0.0

    for start in range(0,dataset_size,batch_size):

        batch_indices=indices[start:start+batch_size]

        batch_states=states_tensor[batch_indices]
        batch_actions=actions_tensor[batch_indices]

        pred_actions=bc_actor_network(batch_states)

        loss=loss_fun(pred_actions,batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print(
        f"Epoch {epoch+1}, Loss:{total_loss:.6f}"
    )

bc_actor_network.eval()
torch.save(
    bc_actor_network.state_dict(),
    "checkpoints/bc_actor.pth"
)
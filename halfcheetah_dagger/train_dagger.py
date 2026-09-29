import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from models import BCActorNetwork

expert_data=np.load(
    "expert_dataset.npz"
)

dagger_data=np.load(
    "dagger_dataset.npz"
)

states=np.concatenate(
    [
        expert_data["states"],
        dagger_data["states"]
    ],
    axis=0
)


actions=np.concatenate(
    [
        expert_data["actions"],
        dagger_data["actions"]
    ],
    axis=0
)

states_tensor=torch.tensor(states,dtype=torch.float32)
actions_tensor=torch.tensor(actions,dtype=torch.float32)

state_dims=17
action_dims=6

dagger_actor_network=BCActorNetwork(state_dims,action_dims)
dagger_actor_network.load_state_dict(
    torch.load(
        "checkpoints/bc_actor.pth",
        weights_only=True
    )
)
dagger_actor_network.train()
optimizer=optim.Adam(
    dagger_actor_network.parameters(),
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

        pred_actions=dagger_actor_network(batch_states)

        loss=loss_fun(pred_actions,batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print(
        f"Epoch {epoch+1}, Loss:{total_loss:.6f}"
    )

dagger_actor_network.eval()
torch.save(
    dagger_actor_network.state_dict(),
    "checkpoints/dagger_actor.pth"
)
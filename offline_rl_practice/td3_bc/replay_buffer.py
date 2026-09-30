import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, data_path):

        data=np.load(data_path)


        self.states=torch.tensor(
            data["states"],
            dtype=torch.float32
        )


        self.actions=torch.tensor(
            data["actions"],
            dtype=torch.float32
        )


        self.rewards=torch.tensor(
            data["rewards"],
            dtype=torch.float32
        ).unsqueeze(1)


        self.next_states=torch.tensor(
            data["next_states"],
            dtype=torch.float32
        )


        self.dones=torch.tensor(
            data["dones"],
            dtype=torch.float32
        ).unsqueeze(1)



        self.size=len(self.states)



    def sample(self,batch_size):

        indices=torch.randint(
            0,
            self.size,
            (batch_size,)
        )


        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

# buffer=ReplayBuffer(
#     "offline_dataset.npz"
# )


# batch=buffer.sample(256) 


# for item in batch:
#     print(item.shape)
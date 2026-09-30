import torch

from replay_buffer import ReplayBuffer
from td3_bc_models import TD3Agent



state_dim=17
action_dim=6


buffer=ReplayBuffer(
    "offline_dataset.npz"
)


agent=TD3Agent(
    state_dim,
    action_dim
)



iterations=50000

batch_size=256



for step in range(iterations):


    states,actions,rewards,next_states,dones = buffer.sample(
        batch_size
    )


    critic1_loss, critic2_loss = agent.update_critic(
        states,
        actions,
        rewards,
        next_states,
        dones
    )


    actor_loss,rl_loss,bc_loss,q = agent.update_actor(
        states,
        actions
    )


    if step % 2 == 0:

        actor_loss,rl_loss,bc_loss,q=agent.update_actor(
            states,
            actions
        )

        agent.soft_update()



    if step % 1000 == 0:

        print(
            f"""
        step:{step}

        Q:{q}

        critic1:{critic1_loss}
        
        critic2:{critic2_loss}

        actor:{actor_loss}

        RL:{rl_loss}

        BC:{bc_loss}
        """
        )

torch.save(
    agent.actor.state_dict(),
    "checkpoints/td3_bc_actor.pth"
)
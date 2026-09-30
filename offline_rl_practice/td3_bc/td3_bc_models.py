import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim
    ):
        super().__init__()


        self.net=nn.Sequential(

            # state_dim -> 256
            nn.Linear(state_dim,256),
            # ReLU
            nn.ReLU(),
            # 256 -> 256
            nn.Linear(256,256),
            # ReLU
            nn.ReLU(),
            # 256 -> action_dim
            nn.Linear(256,action_dim),
            # tanh
            nn.Tanh()

        )


    def forward(self,state):

        return self.net(state)

class Critic(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim
    ):
        super().__init__()


        self.net=nn.Sequential(
            
            # state_dim+action_dim -> 256
            nn.Linear(state_dim+action_dim,256),
            # ReLU
            nn.ReLU(),
            # 256 -> 256
            nn.Linear(256,256),
            # ReLU
            nn.ReLU(),
            # 256 -> 1
            nn.Linear(256,1)

        )


    def forward(
        self,
        state,
        action
    ):


        x=torch.cat(
            [
                state,
                action
            ],
            dim=1
        )


        return self.net(x)

class TD3Agent:

    def __init__(self,state_dim,action_dim):
        
        self.actor=Actor(state_dim,action_dim)
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=1e-4)
        self.actor_target=Actor(state_dim,action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1=Critic(state_dim,action_dim)
        self.critic1_optimizer=optim.Adam(self.critic1.parameters(),lr=1e-4)

        self.critic2=Critic(state_dim,action_dim)
        self.critic2_optimizer=optim.Adam(self.critic2.parameters(),lr=1e-4)


        self.critic1_target=Critic(state_dim,action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target=Critic(state_dim,action_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def update_critic(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
        gamma=0.99
    ):

        with torch.no_grad():

            next_actions = self.actor_target(
                next_states
            )

            noise = torch.randn_like(next_actions)*0.2

            noise=torch.clamp(
                noise,
                -0.5,
                0.5
            )


            next_actions = next_actions + noise

            next_actions=torch.clamp(
                next_actions,
                -1,
                1
            )


            target_q1 = self.critic1_target(
                next_states,
                next_actions
            )


            target_q2 = self.critic2_target(
                next_states,
                next_actions
            )


            target_q=torch.min(
                target_q1,
                target_q2
            )


            target = (
                rewards
                +
                gamma
                *
                (1-dones)
                *
                target_q
            )


        current_q1=self.critic1(
            states,
            actions
        )


        current_q2=self.critic2(
            states,
            actions
        )


        critic1_loss=nn.functional.mse_loss(
            current_q1,
            target
        )


        critic2_loss=nn.functional.mse_loss(
            current_q2,
            target
        )


        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()


        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()


        return (
            critic1_loss.item(),
            critic2_loss.item()
        )

    def update_actor(
        self,
        states,
        actions
    ):

        new_actions=self.actor(
        states
    )

        q_value = self.critic1(
            states,
            new_actions
        )


        lambda_rl = 2.5 / q_value.abs().mean().detach()


        rl_loss = -lambda_rl * q_value.mean()


        bc_loss = nn.functional.mse_loss(
            new_actions,
            actions
        )


        actor_loss = rl_loss + bc_loss


        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()


        return (
            actor_loss.item(),
            rl_loss.item(),
            bc_loss.item(),
            q_value.mean().item()
        )

    def soft_update(self,tau=0.005):

        for param,target_param in zip(
            self.actor.parameters(),
            self.actor_target.parameters()
        ):
            target_param.data.copy_(
                tau*param.data
                +(1-tau)*target_param.data
            )


        for param,target_param in zip(
            self.critic1.parameters(),
            self.critic1_target.parameters()
        ):
            target_param.data.copy_(
                tau*param.data
                +(1-tau)*target_param.data
            )


        for param,target_param in zip(
            self.critic2.parameters(),
            self.critic2_target.parameters()
        ):
            target_param.data.copy_(
                tau*param.data
                +(1-tau)*target_param.data
            )



        
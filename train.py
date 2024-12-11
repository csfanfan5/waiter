import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from typing import TypeVar
from jaxtyping import Float
from env import Restaurant

res1 = Restaurant(
    w=13, 
    h=7, 
    tables=[
        [1, 4, 4, 6],   # Top-left
        [5, 8, 4, 6],   # Top-center
        [9, 12, 4, 6],  # Top-right
        [1, 4, 1, 3],   # Bottom-left
        [5, 8, 1, 3],   # Bottom-center
        [9, 12, 1, 3],  # Bottom-right
    ],
    v=0.1,
    p=0.03)

class ValueNetwork(nn.Module):
    """Evaluating the baseline function V(s,h)."""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 64),  # Concatenate s and h
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output scalar value
        )

    def forward(self, states, timesteps):
        """
        Arguments:
        - states: [B, state_dim]
        - timesteps: [B, 1] (e.g., normalized timestep or raw h)

        Returns:
        - values: [B], predicted value for each (s, h)
        """
        inputs = torch.cat([states, timesteps], dim=-1)  # Concatenate s and h
        return self.net(inputs).squeeze(-1)

    def value_loss(value_net, states, returns):
        """
        Compute the mean squared error between predicted values and observed returns.
        
        Arguments:
        - value_net: The value network (predicts V(s)).
        - states: Batch of states [B, state_dim].
        - returns: Observed returns [B].
        
        Returns:
        - loss: Scalar MSE loss.
        """
        predicted_values = value_net(states)  # Shape: [B]
        return nn.MSELoss()(predicted_values, returns)


class DiscretePolicy(nn.Module):
    """A feedforward neural network for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

    def forward(
        self, states: Float[Tensor, "B state_dim"]
    ) -> Float[Tensor, "B action_dim"]:
        """Returns the action distribution for each state in the batch."""
        logits = self.net(states)
        return logits.float()

class PPO:
    def __init__(self, res: Restaurant, horizon: int):
        # Policy network setup

        # state dim: 5 parameters per table (4 sides + time), position (x,y) of agent, width and height
        self.state_dim = len(res.tables) * 5 + 4
        self.action_dim = 36 # 0, 10, ... 350 degrees

        # group static state variables for efficiency
        self.static_env_tensor = torch.tensor([coord for table in res.tables for coord in table] + [res.w, res.h], dtype=torch.float32)
       
        self.policy = DiscretePolicy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.1)
        self.env = res
        self.H = horizon

    def sample_from_logits(self, logits):
        probs = torch.softmax(logits, dim=-1)

        # sample using multinomial distribution
        action = torch.multinomial(probs, num_samples=1)
        
        return action.item()
    
    def create_batch(self, num_traj):
        """
        Creates a batch of state, value_func pairs.
        Uses policy currently specified by self.policy.
        Used for training the baseline function.
        
        Arguments:
        - num_traj: number of trajectories in this batch
        
        Returns:
        - states: list of tensors
        - value_funcs: list of integers
        """
        states = []
        value_funcs = []
        for _ in range(num_traj):
            rewards = []
            times, agent = self.env.reset()
            for i in range(self.H):
                logits = self.policy.forward()
                action = self.sample_from_logits(logits)

                # get angle alpha to travel in
                alpha = np.radians(10 * action)

                state = torch.cat([
                    self.static_env_tensor,  # Assumes this is already a tensor
                    torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                    torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                    torch.tensor([i], dtype=torch.float32)
                ], dim=0)
                states.append(state)
                times, agent, reward = self.env.step(alpha)
                rewards.append(reward)
            tot_reward = 0
            cumulative = []
            for i in range(num_traj - 1, -1, -1):
                tot_reward += rewards[i]
                cumulative.append(tot_reward)
            for i in range(num_traj - 1, -1, -1):
                value_funcs.append(cumulative[i])
        return states, value_funcs

    def compute_loss(self, new_policy, old_policy, value_predictor, lamb=0.1):
        # rolls out num_traj trajectories
        # now calculate PPO objective, note we flip the sign of the objective to do gradient descent on it
        num_traj = 20 # number of trajectories to average over
        tot_loss = 0
        for _ in num_traj:
            times, agent = self.env.reset()
            for i in range(self.H):
                current_state = torch.cat([
                    self.static_env_tensor,  # Assumes this is already a tensor
                    torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                    torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                    torch.tensor([i], dtype=torch.float32)
                ], dim=0)
                
                old_logits = old_policy.forward()
                old_probs = torch.softmax(old_logits, dim=-1)
                new_logits = new_policy.forward()
                new_probs = torch.softmax(new_logits, dim=-1)
                
                action = self.sample_from_logits(old_logits)

                # get angle alpha to travel in
                alpha = np.radians(10 * action)

                times, agent, reward = self.env.step(alpha)

                next_state = torch.cat([
                    self.static_env_tensor,  # Assumes this is already a tensor
                    torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                    torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                    torch.tensor([i + 1], dtype=torch.float32)
                ], dim=0)

                advantage = reward + value_predictor(next_state) - value_predictor(current_state)

                ratio = new_probs[action] / old_probs[action]

                entropy_term = lamb * np.log(new_probs[action])

                tot_loss -= (ratio * advantage - entropy_term)
        return tot_loss / (num_traj * self.H)


    def one_optimization_step(self, old_policy):
        """
        Performs arg max. I.e. does multiple steps of gradient descent
        """
        #train advantage function neural net for this policy. S x A x H -> R
        # we should have something called advantage in the end
        # create value_net which is V(s,h) baseline prediction for current policy
        value_net = ValueNetwork(self.state_dim)
        value_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)
        num_epochs = 10
        num_batches = 1
        data = []
        for _ in range(num_batches):
            data.append(self.create_batch(num_traj=10))
        for _ in range(num_epochs):
            for batch in data:  # Load batches of states and observed returns
                states, value_funcs = batch

                # Compute the loss
                loss = value_net.value_loss(value_net, states, value_funcs)

                # Update the value network
                value_optimizer.zero_grad()
                loss.backward()
                value_optimizer.step()
        
        # initialize a new policy neural net that is the same as "policy"
        new_policy = DiscretePolicy(state_dim, action_dim)
        new_policy.load_state_dict(old_policy.state_dict())
        # while ...
            # compute loss
            loss = self.compute_loss(new_policy, old_policy, value_net)
            new_policy.optimizer.zero_grad()
            loss.backward()
            new_policy.optimizer.step()
        

        





def ppo(
    env, # instance of class Restaurant
    pi: Callable[[Float[Array, " D"]], Callable[[State, Action], float]],
    λ: float,
    theta_init: Float[Array, " D"],
    n_iters: int,
    n_fit_trajectories: int,
    n_sample_trajectories: int,
):
    theta = theta_init
    for _ in range(n_iters):
        fit_trajectories = sample_trajectories(env, pi(theta), n_fit_trajectories)
        A_hat = fit(fit_trajectories)

        sample_trajectories = sample_trajectories(env, pi(theta), n_sample_trajectories)
        
        def objective(theta_opt):
            total_objective = 0
            for tau in sample_trajectories:
                for s, a, _r in tau:
                    total_objective += pi(theta_opt)(s, a) / pi(theta)(s, a) * A_hat(s, a) + λ * jnp.log(pi(theta_opt)(s, a))
            return total_objective / n_sample_trajectories
        
        theta = optimize(objective, theta)

    return theta

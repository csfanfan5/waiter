import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from env import Restaurant

import argparse 


class ValueNetwork(nn.Module):
    """Evaluating the baseline function V(s,h)."""
    def __init__(self, state_dim):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim + 1, 128),  # Concatenate s and h
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)  # Output scalar value
        # )

        # simplified 
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 16),  # Concatenate s and h
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output scalar value
        )
    def forward(self, states):
        """
        Arguments:
        - states: [B, state_dim]

        Returns:
        - values: [B], predicted value for each (s, h)
        """
        return self.net(states).squeeze(-1)

    def value_loss(self, value_net, states, returns):
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
            nn.Linear(state_dim + 1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )
        for layer in self.net[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Initialize the final layer to produce uniform outputs
        final_layer = self.net[-1]
        nn.init.zeros_(final_layer.weight)  # Set weights to zero
        nn.init.zeros_(final_layer.bias)    # Set biases to zero
    def forward(self, states):
        """Returns the action distribution for each state in the batch."""
        logits = self.net(states)
        return logits.float()

class PPO:
    def __init__(self, res: Restaurant, horizon: int, advantage : bool):
        self.advantage = advantage
        # Policy network setup

        # state dim: 5 parameters per table (4 sides + time), position (x,y) of agent, width and height
        self.state_dim = len(res.tables) * 5 + 4
        self.action_dim = 8

        # group static state variables for efficiency
        self.static_env_tensor = torch.tensor([coord for table in res.tables for coord in table] + [res.w, res.h], dtype=torch.float32)
       
        self.policy = DiscretePolicy(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.env = res
        self.H = horizon

        self.Vepochs = 5
        self.Vbatches = 5
        self.Vbatchsize = 5

        self.Vepochs = 100
        self.Vbatches = 100
        self.Vbatchsize = 5


        self.Pbatches = 5
        self.Pbatchsize = 5

        self.learning_steps = 30

    def sample_from_logits(self, logits):
        probs = torch.softmax(logits, dim=-1)
        if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
            print("Logits:", logits)
            print("Probs:", probs)
            raise ValueError("Logits resulted in invalid probabilities (NaN or Inf).")
        if torch.any(probs < 0):
            print("Logits:", logits)
            print("Probs:", probs)
            raise ValueError("Probs contain negative values.")
        # sample using multinomial distribution
        action = torch.multinomial(probs, num_samples=1)
        
        return action.item()
    
    def create_trajectory(self, policy: DiscretePolicy):
        """
        Rolls out one trajectory according to policy
        """
        # each element is tensor[tables, width, height, times, agent, timestep h]
        states = []
        # num from 0 to 35
        actions = []
        # reward!
        rewards = []
        times, agent = self.env.reset()
        for i in range(self.H):
            state = torch.cat([
                self.static_env_tensor,  # Assumes this is already a tensor
                torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                torch.tensor([i], dtype=torch.float32)
            ], dim=0)
            
            logits = policy.forward(state)
            action = self.sample_from_logits(logits)
            
            # get angle alpha to travel in
            alpha = np.radians((360 / self.action_dim) * action)

            
            times, agent, reward = self.env.step(alpha)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        return states, actions, rewards
        
    
    def rewards_to_value(self, rewards):
        """
        Turns a list of rewards per time step into value functions.
        """
        tot_reward = 0
        values = []
        for i in range(len(rewards) - 1, -1, -1):
            tot_reward += rewards[i]
            values.append(tot_reward)
        values.reverse()
        return values
    
    def create_state_value_batches(self, num_batches, batch_size):
        state_batches = []
        value_batches = []
        for _ in range(num_batches):  # Load batches of states and observed returns   
            state_batch = []   
            value_batch = []    
            for _ in range(batch_size):
                states, _, rewards = self.create_trajectory(self.policy)
                values = self.rewards_to_value(rewards)
                state_batch.extend(states)
                value_batch.extend(values)
            state_batches.append(torch.stack(state_batch))
            value_batches.append(torch.tensor(value_batch, dtype=torch.float32))
        
        return state_batches, value_batches

    def compute_loss(self, new_policy, old_policy, value_predictor=None, lamb=0.1, use_advantage=False):
        # rolls out num_traj trajectories
        # now calculate PPO objective, note we flip the sign of the objective to do gradient descent on it
        tot_loss = 0
        for _ in range(self.Pbatchsize):
            times, agent = self.env.reset()
            for i in range(self.H):
                current_state = torch.cat([
                    self.static_env_tensor,  # Assumes this is already a tensor
                    torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                    torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                    torch.tensor([i], dtype=torch.float32)
                ], dim=0)
                
                old_logits = old_policy.forward(current_state)
                old_probs = torch.softmax(old_logits, dim=-1)
                new_logits = new_policy.forward(current_state)
                new_probs = torch.softmax(new_logits, dim=-1)
                
                action = self.sample_from_logits(old_logits)

                # get angle alpha to travel in
                alpha = np.radians((360 / self.action_dim) * action)

                times, agent, reward = self.env.step(alpha)

                next_state = torch.cat([
                    self.static_env_tensor,  # Assumes this is already a tensor
                    torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                    torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                    torch.tensor([i + 1], dtype=torch.float32)
                ], dim=0)

                if use_advantage:
                    assert value_predictor is not None
                    advantage = reward + value_predictor(next_state) - value_predictor(current_state)
                else: 
                    advantage = 1

                eps = 1e-8  # A small constant to prevent division by zero
                eps_clip = 1e-6
                ratio = (new_probs[action] + eps) / (old_probs[action] + eps)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

                entropy_term = lamb * torch.log(new_probs[action] + eps)

                tot_loss -= (torch.min(surr1, surr2) + entropy_term)

        # return tot_loss / (self.Pbatchsize * self.H)
        return tot_loss / self.Pbatchsize


    def optim_step(self):
        """
        Performs arg max. I.e. does multiple steps of gradient descent
        """
        # train advantage function neural net for this policy. S x A x H -> R
        # we should have something called advantage in the end
        # create value_net which is V(s,h) baseline prediction for current policy
        value_net = ValueNetwork(self.state_dim)
        value_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)

        for _ in range(self.Vepochs):
            state_batches, value_batches = self.create_state_value_batches(self.Vbatches, self.Vbatchsize)
            for i in range(self.Vbatches):
                loss = value_net.value_loss(value_net, state_batches[i], value_batches[i])
                print(loss.item())
                # Update the value network
                value_optimizer.zero_grad()
                loss.backward()
                value_optimizer.step()
        
        # initialize a duplicate that allows us to improve upon self.policy
        objective_sum = 0
        duplicate_policy = DiscretePolicy(self.state_dim, self.action_dim)
        duplicate_policy.load_state_dict(self.policy.state_dict())
        for _ in range(self.Pbatches):
            # compute loss
            loss = self.compute_loss(self.policy, duplicate_policy, value_net, use_advantage=True)
            print(loss)
            objective_sum -= loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        # for the purpose of graphing
        print("Completed one step of optimization!\n")
        return objective_sum / self.Pbatches

    def optim_step_no_advantage(self): 
        """
        Performs arg max. I.e. does multiple steps of gradient descent without reducing 
        variance via an advantage function.
        """
        # initialize a duplicate that allows us to improve upon self.policy
        objective_sum = 0
        duplicate_policy = DiscretePolicy(self.state_dim, self.action_dim)
        duplicate_policy.load_state_dict(self.policy.state_dict())
        for _ in range(self.Pbatches):
            # compute loss
            loss = self.compute_loss(self.policy, duplicate_policy, use_advantage=False, lamb=5.0)
            print(loss)
            objective_sum += loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        # for the purpose of graphing
        return objective_sum / self.Pbatches

    
    def learn(self):
        objectives = []
        for i in range(self.learning_steps):
            if self.advantage:
                avg_objective_val = self.optim_step()
            else:
                avg_objective_val = self.optim_step_no_advantage()
            objectives.append(avg_objective_val)
            print(f"{i} step completed")
        # for the purpose of graphing
        return objectives
    
    def save_nns(self):
        torch.save(self.policy.state_dict(), "model_weights.pth")


def get_args():
    """Parses command line arguments."""

    parser = argparse.ArgumentParser(description="") 

    parser.add_argument(
            ) 

    parser.add_argument(
        "--plots_dir",
        default="./plots",
        help="directory to save plots to",
    )

    # behavioral cloning args
    parser.add_argument(
        "--bc_epochs", type=int, help="number of supervised learning epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--num_dataset_samples",
        type=int,
        help="number of samples to start dataset off with",
    )

    # DAgger args
    parser.add_argument(
        "--num_rollout_steps",
        type=int,
        help="number of steps to roll out with the policy",
    )
    parser.add_argument(
        "--dagger_epochs", type=int, help="number of steps to run dagger"
    )

    args = parser.parse_args()
    return args

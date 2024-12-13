import numpy as np
from typing import Any 
from torch import Tensor
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from env2 import Restaurant


class ValueNetwork(nn.Module):
    """Evaluating the baseline function V(s,h)."""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),  # Concatenate s and h
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output scalar value
        )
        for layer in self.net[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Initialize the final layer to produce uniform outputs
        final_layer=  self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)  # Set weights to zero
            nn.init.zeros_(final_layer.bias)    # Set biases to zero

    def forward(self, states):
        """
        Arguments:
        - states: [B, state_dim]

        Returns:
        - values: [B], predicted value for each (s, h)
        """
        return self.net(states).squeeze(-1)

    def value_loss(self, states, returns):
        """
        Compute the mean squared error between predicted values and observed returns.
        
        Arguments:
        - value_net: The value network (predicts V(s)).
        - states: Batch of states [B, state_dim].
        - returns: Observed returns [B].
        
        Returns:
        - loss: Scalar MSE loss.
        """
        predicted_values = self.net(states)  # Shape: [B]
        return nn.MSELoss()(predicted_values, returns)


class DiscretePolicy(nn.Module):
    """A feedforward neural network for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
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
        final_layer=  self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)  # Set weights to zero
            nn.init.zeros_(final_layer.bias)    # Set biases to zero



    def forward(self, states):
        """Returns the action distribution for each state in the batch."""
        logits = self.net(states)
        return logits.float()

class PPO:
    def __init__(self, res: Restaurant, advantage : bool, 
                 pbatches : int, pbatchsize : int, learning_steps : int, lr : float = 3e-4, lamb : float = 0.1):
        self.advantage = advantage

        # state dim: 5 parameters per table (4 sides + time), position (x,y) of agent, width and height
        # self.state_dim = len(res.tables) * 5 + 4
        self.state_dim = len(res.tables) + 2
        self.action_dim = 4


        self.lr = lr

        # group static state variables for efficiency
        self.static_env_tensor = torch.tensor([coord for table in res.tables for coord in table] + [res.w, res.h], dtype=torch.float32)
       
        self.policy = DiscretePolicy(self.state_dim, self.action_dim)
        self.value_net = ValueNetwork(self.state_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.env = res

        self.Vepochs = 5
        self.Vbatches = 5
        self.Vbatchsize = 5


        self.Pbatches = pbatches
        self.Pbatchsize = pbatchsize
        self.learning_steps = learning_steps

        self.lamb = lamb 

    def sample_from_logits(self, logits):
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1)
        return action.item()
    
    
    def collect_batch(self, batch_size):
        """
        Collect a batch of transitions (states, actions, rewards, old_log_probs) using the current policy.
        """
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_log_probs = []

        for _ in range(batch_size):
            times, agent = self.env.reset()
            done = False 
            time = 0
            while not done:
                # Construct current state
                state = torch.cat([
                    # self.static_env_tensor,
                    torch.tensor(times, dtype=torch.float32),
                    torch.tensor(agent, dtype=torch.float32),
                    # torch.tensor([time], dtype=torch.float32)
                ], dim=0)
                
                logits = self.policy.forward(state)
                probs = torch.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample()

                log_prob = action_dist.log_prob(action)
                alpha = np.radians((360 / self.action_dim) * action.item())

                next_times, next_agent, reward, done = self.env.step(alpha)

                # Store transition
                batch_states.append(state)
                batch_actions.append(action.item())
                batch_rewards.append(reward)
                batch_log_probs.append(log_prob.detach())

                times = next_times
                agent = next_agent
                time += 1

        return (torch.stack(batch_states), 
                torch.tensor(batch_actions), 
                torch.tensor(batch_rewards, dtype=torch.float32), 
                torch.stack(batch_log_probs))

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
        done = False 
        time = 0 
        while not done: 
            state = torch.cat([
                # self.static_env_tensor,  # Assumes this is already a tensor
                torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                # torch.tensor([time], dtype=torch.float32)
            ], dim=0)
            
            logits = policy.forward(state)
            action = self.sample_from_logits(logits)
            
            # get angle alpha to travel in
            alpha = np.radians((360 / self.action_dim) * action)

            times, agent, reward, done = self.env.step(alpha)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            time += 1

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


    def compute_loss(self, new_policy, old_policy, value_predictor, states, actions, rewards, old_log_probs):
        # Compute advantages if needed. For now, just treat reward as advantage placeholder.
        # Proper advantage = reward + gamma*V(next_state) - V(current_state) should be computed,
        # but we omit that for the sake of focusing on fixes #1 and #6.
        advantages = rewards  # This is a simplification; in practice, compute true advantages.

        logits = new_policy(states)
        new_probs = torch.softmax(logits, dim=-1)

        with torch.no_grad():
            old_logits = old_policy(states)
            old_probs = torch.softmax(old_logits, dim=-1)

        # Compute ratios
        eps = 1e-8
        eps_clip = 0.2  # Typically around 0.1-0.2
        action_indices = actions.view(-1,1)
        # Gather probs for chosen actions
        new_probs_act = new_probs.gather(1, action_indices)
        old_probs_act = old_probs.gather(1, action_indices)

        ratio = (new_probs_act + eps) / (old_probs_act + eps)

        surr1 = ratio * advantages.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages.unsqueeze(1)

        # Entropy (optional)
        entropy = torch.distributions.Categorical(new_probs).entropy().mean()

        loss = -torch.min(surr1, surr2).mean() + self.lamb * entropy
        return loss
    def compute_loss_old(self, new_policy, old_policy, value_predictor=None):
        # rolls out num_traj trajectories
        # now calculate PPO objective, note we flip the sign of the objective to do gradient descent on it
        tot_loss = 0
        for _ in range(self.Pbatchsize):
            times, agent = self.env.reset()
            time = 0 
            done = False 
            while not done: 
                current_state = torch.cat([
                    self.static_env_tensor,  # Assumes this is already a tensor
                    torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                    torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                    torch.tensor([time], dtype=torch.float32)
                ], dim=0)
                
                old_logits = old_policy.forward(current_state)
                old_probs = torch.softmax(old_logits, dim=-1)
                new_logits = new_policy.forward(current_state)
                new_probs = torch.softmax(new_logits, dim=-1)
                
                action = self.sample_from_logits(old_logits)

                # get angle alpha to travel in
                alpha = np.radians((360 / self.action_dim) * action)

                times, agent, reward, done = self.env.step(alpha)

                next_state = torch.cat([
                    self.static_env_tensor,  # Assumes this is already a tensor
                    torch.tensor(times, dtype=torch.float32),  # Convert list to tensor
                    torch.tensor(agent, dtype=torch.float32),   # Convert list to tensor
                    torch.tensor([time + 1], dtype=torch.float32)
                ], dim=0)

                if self.advantage:
                    assert value_predictor is not None
                    advantage = reward + value_predictor(next_state) - value_predictor(current_state)
                else: 
                    advantage = 1

                eps = 1e-8  # A small constant to prevent division by zero
                eps_clip = 1e-6
                ratio = (new_probs[action] + eps) / (old_probs[action] + eps)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

                entropy_term = -1 * self.lamb * torch.log(new_probs[action] + eps)

                tot_loss -= (torch.min(surr1, surr2) + entropy_term)

        return tot_loss / self.Pbatchsize



    def optim_step(self):
        # First, collect a batch of data using the current policy (this will serve as old data)
        # For instance, collect one batch of size self.Pbatchsize (number of trajectories)
        states, actions, rewards, old_log_probs = self.collect_batch(batch_size=self.Pbatchsize)

        # Compute returns or advantages here if needed
        # For now, we just use returns directly, as previously defined.
        # Proper advantage estimation is recommended but omitted here.
        values = self.rewards_to_value(rewards)
        values = torch.tensor(values, dtype=torch.float32)
        values = values.unsqueeze(1)

        # Create old_policy as a fixed snapshot of the current policy
        old_policy = DiscretePolicy(self.state_dim, self.action_dim)
        old_policy.load_state_dict(self.policy.state_dict())
        old_policy.eval()  # ensure old_policy is not trained

        # Create and train value network
        value_optimizer = self.value_optimizer
        value_net = self.value_net 

        # Train value network on the collected data (simple approach)
        # Here we do just one step for demonstration. In practice, do multiple epochs.

        # Now optimize the policy multiple epochs on the same batch of data
        objective_sum = 0
        for _ in range(self.Pbatches):
            # Compute PPO loss on the fixed batch
            loss = self.compute_loss(self.policy, old_policy, value_net, states, actions, rewards, old_log_probs)
            objective_sum -= loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return objective_sum / self.Pbatches



    def optim_step_old(self):
        """
        Performs arg max. I.e. does multiple steps of gradient descent
        """
        # train advantage function neural net for this policy. S x A x H -> R
        # we should have something called advantage in the end
        # create value_net which is V(s,h) baseline prediction for current policy
        value_net = ValueNetwork(self.state_dim)
        value_optimizer = optim.Adam(value_net.parameters(), lr=self.lr)

        # for _ in range(self.Vepochs):
        #     for i in range(self.Vbatches):

        # initialize a duplicate that allows us to improve upon self.policy
        objective_sum = 0
        duplicate_policy = DiscretePolicy(self.state_dim, self.action_dim)
        duplicate_policy.load_state_dict(self.policy.state_dict())
        for i in range(self.Pbatches):
            state_batches, value_batches = self.create_state_value_batches(self.Vbatches, self.Vbatchsize)
            vloss = value_net.value_loss(state_batches[i], value_batches[i].unsqueeze(1))
            # Update the value network
            value_optimizer.zero_grad()
            vloss.backward()
            value_optimizer.step()

            # compute loss
            loss = self.compute_loss(self.policy, duplicate_policy, value_net)
            print(loss)
            objective_sum -= loss
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

        # for the purpose of graphing
        return objective_sum / self.Pbatches

    # def optim_step_no_advantage(self): 
    #     """
    #     Performs arg max. I.e. does multiple steps of gradient descent without reducing 
    #     variance via an advantage function.
    #     """
    #     # initialize a duplicate that allows us to improve upon self.policy
    #     objective_sum = 0
    #     duplicate_policy = DiscretePolicy(self.state_dim, self.action_dim)
    #     duplicate_policy.load_state_dict(self.policy.state_dict())
    #     for _ in range(self.Pbatches):
    #         # compute loss
    #         loss = self.compute_loss(self.policy, duplicate_policy)
    #         objective_sum += loss
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #
    #         self.optimizer.step()
    #
    #     # for the purpose of graphing
    #     return objective_sum / self.Pbatches

    
    def learn(self):
        objectives = []
        for _ in tqdm(range(self.learning_steps)):
            if self.advantage:
                avg_objective_val = self.optim_step()
            else:
                avg_objective_val = self.optim_step_no_advantage()
                objectives.append(avg_objective_val)

        # for the purpose of graphing
        return objectives
    
    def save_nns(self, save_path : str):
        torch.save(self.policy.state_dict(), save_path)

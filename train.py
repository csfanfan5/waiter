import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from env import Restaurant


class ValueNetwork(nn.Module):
   """Evaluating the baseline function V(s,h)."""
   def __init__(self, state_dim):
       super().__init__()
       self.net = nn.Sequential(
           nn.Linear(state_dim + 1, 32),  # Concatenate s and h
           nn.ReLU(),
           nn.Linear(32, 32),
           nn.ReLU(),
           nn.Linear(32, 1)  # Output scalar value
       )


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
            nn.Linear(state_dim + 1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, action_dim),
        )
        for layer in self.net[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Initialize the final layer to produce uniform outputs
        final_layer = self.net[-1]
        nn.init.zeros_(final_layer.weight)  # Set weights to zero
        nn.init.zeros_(final_layer.bias)   # Set biases to zero


    def forward(self, states):
        """Returns the action distribution for each state in the batch."""
        logits = self.net(states)
        return logits.float()


class PPO:
    def __init__(self, res: Restaurant, horizon: int):
        # Policy network setup


        # state dim: 5 parameters per table (4 sides + time), position (x,y) of agent, width and height
        self.state_dim = len(res.tables) * 5 + 4
        self.action_dim = 8


        # group static state variables for efficiency
        self.static_env_tensor = torch.tensor([coord for table in res.tables for coord in table] + [res.w, res.h], dtype=torch.float32)

        # value network
        self.value_net = ValueNetwork(self.state_dim)
        self.value_net.load_state_dict(torch.load("value_net.pth"))
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=2e-4)
        self.value_optimizer.load_state_dict(torch.load("value_net_op.pth"))
        
        self.policy = DiscretePolicy(self.state_dim, self.action_dim)
        self.policy.load_state_dict(torch.load("model_weights.pth"))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-3)
        self.optimizer.load_state_dict(torch.load("model_weights_op.pth"))
        
        self.env = res
        self.H = horizon


        #hyperparameters
        self.Vepochs = 5
        self.Vbatches = 10
        self.Vbatchsize = 10


        self.Psteps = 50
        self.Pnum_trajs = 30


        self.learning_steps = 10


    def sample_from_logits(self, logits):
        probs = torch.softmax(logits, dim=-1)
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
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            prob_of_action = probs[action].item()
            
            # get angle alpha to travel in
            alpha = np.radians((360 / self.action_dim) * action)


            
            times, agent, reward = self.env.step(alpha)


            states.append(state)
            actions.append((action, prob_of_action))
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
        """
        Creates state, value_func pairs (more optimal than create_trajs)
        """
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
    
    def create_trajs(self, num_trajs, policy):
        """
        Collects trajectory data for num_trajs
        """
        current_batch = []
        for _ in range(num_trajs):
            traj_data = self.create_trajectory(policy)
            current_batch.append(traj_data)
        return current_batch


    def compute_loss(self, new_policy, trajectories, value_predictor, lamb=1):
        """
        Computes the loss (negative objective) for PPO
        """
        # rolls out num_traj trajectories
        # now calculate PPO objective, note we flip the sign of the objective to do gradient descent on it
        tot_entropy = 0
        tot_loss = 0
        for traj_data in trajectories:
            states, actions, rewards = traj_data
            for i in range(self.H - 1):
                current_state = states[i]
                action, prob_of_action = actions[i] # action taken and probability of that action being taken in old policy
                reward = rewards[i]
                new_logits = new_policy.forward(current_state)
                new_probs = torch.softmax(new_logits, dim=-1)
                next_state = states[i+1]
                advantage = reward + value_predictor(next_state) - value_predictor(current_state)
                eps = 1e-8  # A small constant to prevent division by zero
                eps_clip = 0.2
                ratio = (new_probs[action] + eps) / (prob_of_action + eps)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                #KL = -lamb * torch.log(new_probs[action] + eps)
                entropy = -lamb * torch.sum(new_probs * torch.log(new_probs + eps))
                tot_entropy += entropy
                loss = -torch.min(surr1, surr2) - entropy
                tot_loss += loss
        avg_loss = tot_loss / (self.Pnum_trajs * (self.H - 1))
        avg_entr_loss = tot_entropy / (self.Pnum_trajs * (self.H - 1))
        print("Avg loss: ", avg_loss.item())
        print("Avg entropy loss: ", avg_entr_loss.item())
        return avg_loss


    def optim_step(self):
        """
        Each optimization step of PPO.
        """
        # train value function
        for _ in range(self.Vepochs):
            state_batches, value_batches = self.create_state_value_batches(self.Vbatches, self.Vbatchsize)
            for i in range(self.Vbatches):
                loss = self.value_net.value_loss(state_batches[i], value_batches[i])
                print("Valuefunc loss: ", loss.item())
                # Update the value network
                self.value_optimizer.zero_grad()
                loss.backward()
                self.value_optimizer.step()
                torch.save(self.value_net.state_dict(), "value_net.pth")
                torch.save(self.value_optimizer.state_dict(), "value_net_op.pth")
        
        # optimize self.policy with respect to the loss
        objective_sum = 0
        trajectories = self.create_trajs(self.Pnum_trajs, self.policy)
        for _ in range(self.Psteps):
            loss = self.compute_loss(self.policy, trajectories, self.value_net)
            objective_sum -= loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # for the purpose of graphing
        torch.save(self.policy.state_dict(), "model_weights.pth")
        torch.save(self.optimizer.state_dict(), "model_weights_op.pth")
        
        return objective_sum / self.Psteps
    
    def learn(self):
        objectives = []
        for i in range(self.learning_steps):
            print("Completed step ", i, " of optimization!\n")
            avg_objective_val = self.optim_step()
            objectives.append(avg_objective_val)
        # for the purpose of graphing
        return objectives
    
    def save_nns(self):
        torch.save(self.policy.state_dict(), "model_weights.pth")

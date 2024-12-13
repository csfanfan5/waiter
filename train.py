import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ValueNetwork(nn.Module):
    """Value function approximator: V(s)"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, states):
        return self.net(states).squeeze(-1)

    def value_loss(self, states, returns):
        predicted_values = self.forward(states)
        return nn.MSELoss()(predicted_values, returns)


class DiscretePolicy(nn.Module):
    """Policy network for discrete action spaces."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

    def forward(self, states):
        logits = self.net(states)
        return logits

    def get_action(self, state):
        """Sample action and return log_prob."""
        logits = self.forward(state.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)
        log_prob = torch.log(probs.gather(-1, action))
        return action.item(), log_prob.squeeze(0)

    def get_log_probs(self, states, actions):
        """Get log probabilities of given actions under current policy."""
        logits = self.forward(states)
        probs = torch.softmax(logits, dim=-1)
        return torch.log(probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1))


class PPO:
    def __init__(self, env, horizon=200, gamma=1, eps_clip=0.2, lamb=10, lr=3e-4, vf_lr=1e-3, 
                 update_epochs=25, mini_batch_size=32):
        self.env = env
        # State includes tables plus [width, height] plus [current_time_step], etc.
        # Modify as needed. Originally: len(res.tables)*5 + 5 for state_dim (adjust if necessary)
        self.state_dim = len(env.tables)*5 + 5
        self.action_dim = len(env.tables)  # Number of tables = number of discrete actions
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coeff = lamb
        self.horizon = horizon
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

        # Policy and value networks
        self.policy = DiscretePolicy(self.state_dim, self.action_dim)
        self.value_net = ValueNetwork(self.state_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=vf_lr)

        # For convenience
        self.static_env_tensor = torch.tensor([coord for table in env.tables for coord in table] + [env.w, env.h], dtype=torch.float32)

    def _get_state(self, times, agent, t):
        # Construct the current state vector
        return torch.cat([
            self.static_env_tensor,
            torch.tensor(times, dtype=torch.float32), # times
            torch.tensor(agent, dtype=torch.float32), # agent pos
            torch.tensor([t], dtype=torch.float32)     # current time step
        ], dim=0)

    def collect_trajectories(self, num_trajectories=3):
        # Collect data using the current policy without updating it.
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []

        for _ in range(num_trajectories):
            times, agent = self.env.reset()
            for t in range(self.horizon):
                state = self._get_state(times, agent, t)
                value = self.value_net(state.unsqueeze(0)).item()

                action, log_prob = self.policy.get_action(state)

                # Compute angle towards the chosen table
                table = self.env.tables[action]
                center_x = (table[0] + table[1]) / 2.0
                center_y = (table[2] + table[3]) / 2.0
                dx = center_x - agent[0]
                dy = center_y - agent[1]
                alpha = np.arctan2(dy, dx)

                times, agent, reward = self.env.step(alpha)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        values = torch.tensor(values, dtype=torch.float32)

        # Compute returns and advantages for all trajectories
        returns, advantages = self.compute_advantages(rewards, values, num_trajectories)

        return states, actions, rewards, log_probs, returns, advantages

    def compute_advantages(self, rewards, values, num_trajectories):
        returns_list = []
        advantages_list = []

        for i in range(num_trajectories):
            start = i * self.horizon
            end = (i + 1) * self.horizon

            traj_rewards = rewards[start:end]
            traj_values = values[start:end]

            # Compute returns for this trajectory
            G = 0
            traj_returns = []
            for r in reversed(traj_rewards):
                G = r + self.gamma * G
                traj_returns.append(G)
            traj_returns.reverse()
            traj_returns = torch.tensor(traj_returns, dtype=torch.float32)

            # Compute advantages for this trajectory
            traj_advantages = traj_returns - traj_values

            returns_list.append(traj_returns)
            advantages_list.append(traj_advantages)

        # Concatenate all trajectories
        all_returns = torch.cat(returns_list)
        all_advantages = torch.cat(advantages_list)

        # Normalize advantages across all trajectories
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        return all_returns, all_advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        dataset = torch.utils.data.TensorDataset(states, actions, log_probs_old, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True)

        value_losses = []
        policy_losses = []
        entropies = []
        for _ in range(self.update_epochs):
            for s_batch, a_batch, old_lp_batch, ret_batch, adv_batch in loader:
                new_log_probs = self.policy.get_log_probs(s_batch, a_batch)
                ratio = torch.exp(new_log_probs - old_lp_batch)

                # Clipped surrogate objective
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                logits = self.policy.net(s_batch)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                entropies.append(entropy)
                policy_loss -= self.entropy_coeff * entropy

                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Value loss
                value_loss = self.value_net.value_loss(s_batch, ret_batch)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
        
        print(f"VALUE MEAN: {torch.mean(torch.tensor(value_losses))}")
        print(f"POLICY MEAN: {torch.mean(torch.tensor(policy_losses))}")
        print(f"ENTROPY MEAN: {torch.mean(torch.tensor(entropies))}")

    def train(self, num_iterations=15, num_trajectories=3):
        for i in range(num_iterations):
            with torch.no_grad():
                states, actions, rewards, log_probs, returns, advantages = self.collect_trajectories(num_trajectories)

            nn.utils.clip_grad_norm_(self.policy.parameters(), 100)
            self.update(states, actions, log_probs, returns, advantages)
            print(f"Iteration {i+1}/{num_iterations} completed.")
        
    def save_nns(self):
        torch.save(self.policy.state_dict(), "model_policy.pth")
        torch.save(self.value_net.state_dict(), "model_value.pth")

    def load_nns(self, policy_path="model_policy.pth", value_path="model_value.pth"):
        self.policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
        self.value_net.load_state_dict(torch.load(value_path, map_location=torch.device('cpu')))
        self.policy.eval()
        self.value_net.eval()

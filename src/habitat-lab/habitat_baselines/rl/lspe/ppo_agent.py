import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from .lspe_models import BisimulationEncoder

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

class PPO(nn.Module):
    def __init__(self, action_space, latent_dim=128):
        super(PPO, self).__init__()
        # PPO uses its own encoder to ensure stability
        self.visual_encoder = BisimulationEncoder((3, 256, 256), latent_dim)
        self.action_layer = nn.Linear(latent_dim, action_space)
        self.value_layer = nn.Linear(latent_dim, 1)

    def forward(self, state):
        embed = self.visual_encoder(state)
        return torch.softmax(self.action_layer(embed), dim=-1), self.value_layer(embed)

    def act(self, state):
        probs, val = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        probs, val = self.forward(state)
        dist = Categorical(probs)
        return dist.log_prob(action), val, dist.entropy()

def ppo_update_step(memory, policy, optimizer, gamma=0.99, K_epochs=4, eps_clip=0.2, device="cuda"):
    """Performs one PPO update based on collected memory"""
    if len(memory.states) == 0: return
    
    # Convert list to tensor
    old_states = torch.stack(memory.states).to(device).detach()
    old_actions = torch.tensor(memory.actions).to(device).detach()
    old_logprobs = torch.tensor(memory.logprobs).to(device).detach()
    rewards = torch.tensor(memory.rewards).to(device).detach()
    dones = memory.dones
    
    # Monte Carlo estimate of returns
    returns = []
    discounted_sum = 0
    for reward, is_done in zip(reversed(rewards.tolist()), reversed(dones)):
        if is_done:
            discounted_sum = 0
        discounted_sum = reward + (gamma * discounted_sum)
        returns.insert(0, discounted_sum)
        
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-7)
    
    # Optimize policy for K epochs
    for _ in range(K_epochs):
        # Evaluating old actions and values
        logprobs, state_values, dist_entropy = policy.evaluate(old_states, old_actions)
        state_values = state_values.squeeze()
        
        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - old_logprobs)

        # Finding Surrogate Loss
        advantages = returns - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages

        # Final loss
        loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.functional.mse_loss(state_values, returns) - 0.01 * dist_entropy.mean()
        
        # Take gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
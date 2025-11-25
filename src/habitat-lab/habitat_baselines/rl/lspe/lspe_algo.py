import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from .lspe_models import BisimulationEncoder, DiffusionPredictor

class LSPEAlgo:
    def __init__(self, obs_shape, latent_dim=128, k_horizon=3, lr=1e-4, device="cuda"):
        self.device = device
        self.latent_dim = latent_dim
        self.k_horizon = k_horizon
        
        self.encoder = BisimulationEncoder(obs_shape, latent_dim).to(device)
        self.predictor = DiffusionPredictor(latent_dim).to(device)
        self.target_encoder = copy.deepcopy(self.encoder).to(device)
        
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=lr
        )
        self.current_direction_z = None
        self.reset_direction()

    def reset_direction(self):
        """Sample new random exploration direction"""
        z = torch.randn(self.latent_dim, device=self.device)
        self.current_direction_z = F.normalize(z, p=2, dim=0)

    def compute_intrinsic_reward(self, obs_curr, obs_next, num_samples=5):
        """
        Calculate Reward: <Delta_t, z>^2 * Variance
        """
        with torch.no_grad():
            z_curr = self.encoder(obs_curr)
            z_next = self.encoder(obs_next)
            
            # 1. Latent Displacement
            delta_t = z_next - z_curr
            delta_proj = (delta_t * self.current_direction_z).sum(dim=-1, keepdim=True).pow(2)
            
            # 2. Predictive Uncertainty
            preds = self.predictor.sample(z_curr, num_samples=num_samples)
            preds_proj = (preds * self.current_direction_z).sum(dim=-1)
            variance = preds_proj.var(dim=0, keepdim=True).T
            
            return delta_proj * variance

    def update(self, s_curr, s_next, s_future, r_curr, r_next, gamma=0.99):
        """
        Auxiliary Task Update: Diffusion Loss + Bisimulation Loss
        """
        # A. Diffusion Loss
        z_curr_pred = self.encoder(s_curr)
        with torch.no_grad():
            z_future_target = self.target_encoder(s_future)
            
        t_steps = torch.randint(0, self.predictor.num_timesteps, (s_curr.shape[0],), device=self.device)
        z_noisy, noise = self.predictor.forward_diffusion_sample(z_future_target, t_steps)
        pred_noise = self.predictor.predict_noise(z_noisy, z_curr_pred, t_steps)
        loss_diff = F.mse_loss(pred_noise, noise)
        
        # B. Bisimulation Loss
        # Split batch for pairs
        half = s_curr.shape[0] // 2
        z_i, z_j = z_curr_pred[:half], z_curr_pred[half:]
        dist_ij = (z_i - z_j).abs().sum(dim=-1)
        
        r_diff = torch.abs(r_curr[:half] - r_curr[half:]).squeeze()
        with torch.no_grad():
            z_next_i = self.target_encoder(s_next[:half])
            z_next_j = self.target_encoder(s_next[half:])
            dist_next = (z_next_i - z_next_j).abs().sum(dim=-1)
            target = r_diff + gamma * dist_next
            
        loss_bisim = F.mse_loss(dist_ij, target)
        
        total_loss = loss_diff + loss_bisim
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Soft update target encoder
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)
            
        return total_loss.item()
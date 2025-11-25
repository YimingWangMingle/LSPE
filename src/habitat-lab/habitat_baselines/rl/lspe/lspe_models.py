import torch
import torch.nn as nn
import numpy as np

class BisimulationEncoder(nn.Module):
    """
    LSPE State Encoder: Maps high-dim observations (images) to latent space Z.
    """
    def __init__(self, obs_shape, latent_dim=128):
        super().__init__()
        # Input: (C, H, W) e.g., (3, 256, 256)
        self.net = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            # Based on 256x256 input -> feature map approx 28x28
            nn.Linear(64 * 28 * 28, 512), 
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class DiffusionPredictor(nn.Module):
    """
    LSPE Diffusion-Based Self-Predictive Network (D-SPN).
    Predicts future latent state distribution.
    """
    def __init__(self, latent_dim, num_timesteps=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        
        # Register diffusion buffers
        self.register_buffer('beta', torch.linspace(0.0001, 0.02, num_timesteps))
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16)
        )
        # Noise prediction network
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2 + 16, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )

    def forward_diffusion_sample(self, z_future, t):
        """Forward process: q(z_n | z_0)"""
        noise = torch.randn_like(z_future)
        alpha_bar_t = self.alpha_bar[t][:, None]
        z_noisy = torch.sqrt(alpha_bar_t) * z_future + torch.sqrt(1 - alpha_bar_t) * noise
        return z_noisy, noise

    def predict_noise(self, z_noisy, z_current, t_idx):
        """Reverse process: Predict noise"""
        t_in = t_idx.float().view(-1, 1) / self.num_timesteps
        t_emb = self.time_mlp(t_in)
        inp = torch.cat([z_noisy, z_current, t_emb], dim=-1)
        return self.model(inp)

    @torch.no_grad()
    def sample(self, z_current, num_samples=1):
        """Sampling for uncertainty estimation"""
        B = z_current.shape[0]
        device = z_current.device
        z_curr_exp = z_current.unsqueeze(0).expand(num_samples, -1, -1).reshape(-1, self.latent_dim)
        z = torch.randn_like(z_curr_exp)
        
        for i in reversed(range(self.num_timesteps)):
            t_idx = torch.full((z.shape[0],), i, device=device, dtype=torch.long)
            pred_noise = self.predict_noise(z, z_curr_exp, t_idx)
            alpha, alpha_bar, beta = self.alpha[i], self.alpha_bar[i], self.beta[i]
            noise = torch.randn_like(z) if i > 0 else 0
            
            z = (1 / torch.sqrt(alpha)) * (z - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * noise
            
        return z.view(num_samples, B, self.latent_dim)
# Description: Beta-VAE model for the latent space representation only

import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.nn import functional as F

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "identity":
        return nn.Identity()
    else:
        print("invalid activation function!")
        return None

# Xavier initialization function
def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class VAE(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 kld_weight = 1.0,
                 activation = 'elu',
                 decoder_hidden_dims = [32, 64, 128, 256],
                 encoder_hidden_dims = [256, 256, 256],
                 device='cpu'):
        super(VAE, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent 
        self.device = device

        # Build Encoder
        self.encoder = MLPHistoryEncoder(
            num_obs = num_obs,
            num_history = num_history,
            num_latent = num_latent * 4,
            activation = activation,
            adaptation_module_branch_hidden_dims = encoder_hidden_dims,
        )
        self.latent_mu = nn.Linear(num_latent * 4, num_latent)
        self.latent_var = nn.Linear(num_latent * 4, num_latent)
        self.kld_weight = kld_weight

        # Build Decoder
        modules = []
        activation_fn = get_activation(activation)
        decoder_input_dim = num_latent
        modules.extend([
            nn.Linear(decoder_input_dim, decoder_hidden_dims[0]),
            activation_fn
        ])
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                modules.append(nn.Linear(decoder_hidden_dims[l], num_obs))
            else:
                modules.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                modules.append(activation_fn)
        self.decoder = nn.Sequential(*modules)

        # Apply Xavier initialization
        self.apply(init_weights_xavier)

    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        return [latent_mu, latent_var]

    def decode(self, z):
        output = self.decoder(z)
        return output

    def forward(self, obs_history):
        latent_mu, latent_var = self.encode(obs_history)
        z = self.reparameterize(latent_mu, latent_var)
        return z, [latent_mu, latent_var]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_fn(self, obs_history, next_obs):
        z, latent_params = self.forward(obs_history)
        latent_mu, latent_var = latent_params 
        latent_mu = torch.clamp(latent_mu, min=-1e3, max=1e3)
        latent_var = torch.clamp(latent_var, min=-1e3, max=1e3)

        # Reconstruction loss
        recons = self.decode(z)
        recons = torch.clamp(recons, min=-1e3, max=1e3)
        recons_loss = F.smooth_l1_loss(recons, next_obs, reduction='none').mean(-1)
        # recons_loss = F.mse_loss(recons, next_obs, reduction='none').mean(-1)

        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=1)

        # Handle NaN or Inf
        if (torch.isnan(recons_loss).any() | torch.isinf(recons_loss).any()):
            print('nan')
            recons_loss = torch.nan_to_num(recons_loss, nan=0.0, posinf=1e3, neginf=-1e3)
        if (torch.isnan(kld_loss).any() | torch.isinf(kld_loss).any()):
            print('nan')
            kld_loss = torch.nan_to_num(kld_loss, nan=0.0, posinf=1e3, neginf=-1e3)

        loss = recons_loss + self.kld_weight * kld_loss
        loss = torch.clamp(loss, max=1e4)

        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'kld_loss': kld_loss,
        }

    def sample(self, obs_history):
        estimation, _ = self.forward(obs_history)
        return estimation

    def inference(self, obs_history):
        _, latent_params = self.forward(obs_history)
        latent_mu, latent_var = latent_params
        return latent_mu


class MLPHistoryEncoder(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 adaptation_module_branch_hidden_dims = [256, 128],):
        super(MLPHistoryEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent

        input_size = num_obs * num_history
        output_size = num_latent

        activation = get_activation(activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(input_size, adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(adaptation_module_branch_hidden_dims)):
            if l == len(adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l], output_size))
            else:
                adaptation_module_layers.append(
                    nn.Linear(adaptation_module_branch_hidden_dims[l],
                              adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.encoder = nn.Sequential(*adaptation_module_layers)
    def forward(self, obs_history):
        """
        obs_history.shape = (bz, T , obs_dim)
        """
        bs = obs_history.shape[0]
        T = self.num_history
        output = self.encoder(obs_history.reshape(bs, -1))
        return output


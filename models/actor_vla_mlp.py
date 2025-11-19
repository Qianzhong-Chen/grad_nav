import torch
import torch.nn as nn
from torch.distributions import Normal
from models import model_utils


class TaskAwareActor(nn.Module):
    def __init__(self, state_obs_dim, vlm_dim, action_dim, cfg_network, device='cuda:0'):
        super(TaskAwareActor, self).__init__()

        self.device = device
        self.action_dim = action_dim

        # === state encoder ===
        state_units = cfg_network['actor_mlp']['obs_units']
        state_layers = []
        layer_dims = [state_obs_dim] + state_units
        for i in range(len(layer_dims) - 1):
            state_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            state_layers.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
            state_layers.append(nn.LayerNorm(layer_dims[i + 1]))
        self.state_encoder = nn.Sequential(*state_layers).to(device)

        # === vlm encoder ===
        vlm_units = cfg_network['actor_mlp']['vlm_units']
        vlm_layers = []
        layer_dims = [vlm_dim] + vlm_units
        for i in range(len(layer_dims) - 1):
            vlm_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            vlm_layers.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
            vlm_layers.append(nn.LayerNorm(layer_dims[i + 1]))
        self.vlm_encoder = nn.Sequential(*vlm_layers).to(device)

        # === fusion net ===
        fusion_units = cfg_network['actor_mlp']['fusion_units']
        fusion_layers = []
        layer_dims = [state_units[-1] + vlm_units[-1]] + fusion_units + [action_dim]
        for i in range(len(layer_dims) - 1):
            fusion_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                fusion_layers.append(model_utils.get_activation_func(cfg_network['actor_mlp']['activation']))
                fusion_layers.append(nn.LayerNorm(layer_dims[i + 1]))
            else:
                fusion_layers.append(model_utils.get_activation_func('identity'))
        self.mu_net = nn.Sequential(*fusion_layers).to(device)

        # === logstd ===
        logstd = cfg_network.get('actor_logstd_init', -1.0)
        self.logstd = nn.Parameter(torch.ones(action_dim, dtype=torch.float32, device=device) * logstd)

        print(self.state_encoder)
        print(self.vlm_encoder)
        print(self.mu_net)
        print("logstd:", self.logstd)

    def get_logstd(self):
        return self.logstd

    def forward(self, state_obs, vlm_feature, deterministic=False):
        # vlm_feature = vlm_feature.detach()  # Make sure it doesn't carry autograd graph

        obs_feat = self.state_encoder(state_obs)
        vlm_feat = self.vlm_encoder(vlm_feature)
        fused = torch.cat([obs_feat, vlm_feat], dim=-1)
        mu = self.mu_net(fused)

        if deterministic:
            return mu
        else:
            std = self.logstd.exp().expand_as(mu)
            dist = Normal(mu, std)
            return dist.rsample()

    def forward_with_dist(self, state_obs, vlm_feature, deterministic=False):
        # vlm_feature = vlm_feature.detach()

        obs_feat = self.state_encoder(state_obs)
        vlm_feat = self.vlm_encoder(vlm_feature)
        fused = torch.cat([obs_feat, vlm_feat], dim=-1)
        mu = self.mu_net(fused)
        std = self.logstd.exp().expand_as(mu)

        if deterministic:
            return mu, mu, std
        else:
            dist = Normal(mu, std)
            sample = dist.rsample()
            return sample, mu, std

    def evaluate_actions_log_probs(self, state_obs, vlm_feature, actions):
        vlm_feature = vlm_feature.detach()

        obs_feat = self.state_encoder(state_obs)
        vlm_feat = self.vlm_encoder(vlm_feature)
        fused = torch.cat([obs_feat, vlm_feat], dim=-1)
        mu = self.mu_net(fused)
        std = self.logstd.exp().expand_as(mu)
        dist = Normal(mu, std)

        return dist.log_prob(actions)

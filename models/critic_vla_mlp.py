import torch
import torch.nn as nn
from models import model_utils

class TaskAwareCriticMLP(nn.Module):
    def __init__(self, state_obs_dim, vlm_dim, cfg_network, device='cuda:0'):
        super(TaskAwareCriticMLP, self).__init__()
        self.device = device

        # === state encoder ===
        state_units = cfg_network['critic_mlp']['obs_units']  # e.g., [128, 128]
        state_layers = []
        layer_dims = [state_obs_dim] + state_units
        for i in range(len(layer_dims) - 1):
            state_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            state_layers.append(model_utils.get_activation_func(cfg_network['critic_mlp']['activation']))
            state_layers.append(nn.LayerNorm(layer_dims[i + 1]))
        self.state_encoder = nn.Sequential(*state_layers).to(device)

        # === vlm encoder ===
        vlm_units = cfg_network['critic_mlp']['vlm_units']  # e.g., [64]
        vlm_layers = []
        layer_dims = [vlm_dim] + vlm_units
        for i in range(len(layer_dims) - 1):
            vlm_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            vlm_layers.append(model_utils.get_activation_func(cfg_network['critic_mlp']['activation']))
            vlm_layers.append(nn.LayerNorm(layer_dims[i + 1]))
        self.vlm_encoder = nn.Sequential(*vlm_layers).to(device)

        # === fusion net ===
        fusion_units = cfg_network['critic_mlp']['fusion_units']  # e.g., [128]
        fusion_layers = []
        layer_dims = [state_units[-1] + vlm_units[-1]] + fusion_units + [1]
        for i in range(len(layer_dims) - 1):
            fusion_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                fusion_layers.append(model_utils.get_activation_func(cfg_network['critic_mlp']['activation']))
                fusion_layers.append(nn.LayerNorm(layer_dims[i + 1]))
            else:
                fusion_layers.append(model_utils.get_activation_func('identity'))
        self.value_net = nn.Sequential(*fusion_layers).to(device)

        print(self.state_encoder)
        print(self.vlm_encoder)
        print(self.value_net)

    def forward(self, state_obs, vlm_feature):
        """
        state_obs: [B, state_obs_dim]
        vlm_feature: [B, vlm_dim]
        returns: [B, 1]
        """
        vlm_feature = vlm_feature.detach()  # safety: prevent gradient linkage

        state_feat = self.state_encoder(state_obs)
        vlm_feat = self.vlm_encoder(vlm_feature)
        fused = torch.cat([state_feat, vlm_feat], dim=-1)
        value = self.value_net(fused)
        return value

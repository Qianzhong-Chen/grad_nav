import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models import model_utils
from models.moe import moe_auxiliary_loss


class VLMExpert(nn.Module):
    def __init__(self, state_dim, vlm_dim, action_dim, cfg):
        super().__init__()

        # === state encoder ===
        state_units = cfg['obs_units']
        state_layers = []
        dims = [state_dim] + state_units
        for i in range(len(dims)-1):
            state_layers.append(nn.Linear(dims[i], dims[i+1]))
            state_layers.append(model_utils.get_activation_func(cfg['activation']))
            state_layers.append(nn.LayerNorm(dims[i+1]))
        self.state_encoder = nn.Sequential(*state_layers)

        # === vlm encoder ===
        vlm_units = cfg['vlm_units']
        vlm_layers = []
        dims = [vlm_dim] + vlm_units
        for i in range(len(dims)-1):
            vlm_layers.append(nn.Linear(dims[i], dims[i+1]))
            vlm_layers.append(model_utils.get_activation_func(cfg['activation']))
            vlm_layers.append(nn.LayerNorm(dims[i+1]))
        self.vlm_encoder = nn.Sequential(*vlm_layers)

        # === fusion net ===
        fusion_units = cfg['fusion_units']
        in_dim = state_units[-1] + vlm_units[-1]
        dims = [in_dim] + fusion_units + [action_dim]
        fusion_layers = []
        for i in range(len(dims)-1):
            fusion_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                fusion_layers.append(model_utils.get_activation_func(cfg['activation']))
                fusion_layers.append(nn.LayerNorm(dims[i+1]))
        self.mu_net = nn.Sequential(*fusion_layers)

    def forward(self, state_obs, vlm_feature):
        obs_feat = self.state_encoder(state_obs)
        vlm_feat = self.vlm_encoder(vlm_feature)
        fused = torch.cat([obs_feat, vlm_feat], dim=-1)
        return self.mu_net(fused)


class TaskAwareMoEActor(nn.Module):
    def __init__(self, state_obs_dim, vlm_dim, action_dim, cfg_network, device='cuda:0'):
        super().__init__()
        self.device = device
        self.action_dim = action_dim

        # === Experts ===
        self.num_experts = cfg_network['moe']['num_experts']
        expert_cfg = cfg_network['actor_mlp']
        self.experts = nn.ModuleList([
            VLMExpert(state_obs_dim, vlm_dim, action_dim, expert_cfg).to(device)
            for _ in range(self.num_experts)
        ])

        # === GateNet ===
        gate_input_dim = state_obs_dim + vlm_dim
        gate_units = cfg_network['moe']['gate_mlp']['units']
        gate_act = cfg_network['moe']['gate_mlp']['activation']
        gate_layers = []
        dims = [gate_input_dim] + gate_units + [self.num_experts]
        for i in range(len(dims)-1):
            gate_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                gate_layers.append(model_utils.get_activation_func(gate_act))
        self.gate_net = nn.Sequential(*gate_layers).to(device)

        self.top_k = cfg_network['moe']['top_k']
        logstd_init = cfg_network.get('actor_logstd_init', -1.0)
        self.logstd = nn.Parameter(torch.ones(action_dim, device=device) * logstd_init)

    def forward(self, state_obs, vlm_feature, deterministic=False):
        batch_size = state_obs.size(0)

        # === Run each expert ===
        all_outputs = torch.stack([
            expert(state_obs, vlm_feature)
            for expert in self.experts
        ], dim=1)  # [B, E, action_dim]

        # === Compute gate scores ===
        gate_input = torch.cat([state_obs, vlm_feature], dim=-1)
        gate_logits = self.gate_net(gate_input)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # === Top-k gating ===
        top_k_scores, top_k_indices = torch.topk(gate_probs, self.top_k, dim=1)  # [B, k]

        # Gather expert outputs
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k).to(state_obs.device)
        selected = all_outputs[batch_indices, top_k_indices]  # [B, k, action_dim]

        # Weighted sum
        weights = top_k_scores / (top_k_scores.sum(dim=1, keepdim=True) + 1e-9)
        fused_output = (selected * weights.unsqueeze(-1)).sum(dim=1)  # [B, action_dim]

        std = torch.exp(self.logstd)
        dist = Normal(fused_output, std)
        action = fused_output if deterministic else dist.rsample()

        aux_loss = moe_auxiliary_loss(gate_probs, top_k_indices)

        return action, aux_loss, top_k_indices, top_k_scores

    def evaluate_actions_log_probs(self, state_obs, vlm_feature, actions):
        with torch.no_grad():
            out, _, _, _ = self.forward(state_obs, vlm_feature, deterministic=True)
        std = torch.exp(self.logstd)
        dist = Normal(out, std)
        return dist.log_prob(actions)

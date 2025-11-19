import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim, hidden_units, output_dim, activation='relu', dropout=0.0):
    layers = []

    # Input layer
    layers.append(nn.Linear(input_dim, hidden_units[0]))
    layers.append(get_activation(activation))
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    # Hidden layers
    for i in range(len(hidden_units) - 1):
        layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
        layers.append(get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

    # Output layer
    layers.append(nn.Linear(hidden_units[-1], output_dim))

    return nn.Sequential(*layers)

def get_activation(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class MoE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 gate_dim, 
                 cfg_network,
                 boost=False, 
                 stmoe=False):
        super(MoE, self).__init__()
        moe_cfg = cfg_network['moe']
        num_experts = moe_cfg.get('num_experts', 4)
        self.top_k = moe_cfg.get('top_k', 2)
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = moe_cfg.get('dropout', 0.1)
        self.stmoe = stmoe
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Experts: each from input_dim → hidden → output_dim
        expert_hidden = cfg_network['moe']['expert_mlp'].get('units', [128,128])
        expert_activation = cfg_network['moe']['expert_mlp'].get('activation', 'relu')
        self.experts = nn.ModuleList([
            build_mlp(input_dim, expert_hidden, output_dim, expert_activation, self.dropout)
            for _ in range(self.num_experts)
        ])
        self.experts = self.experts.to(self.device)
        self.lambda_entropy = moe_cfg.get('lambda_entropy', 0.1)
        self.lambda_balance = moe_cfg.get('lambda_balance', 5.0)

        # Gate: input_dim → hidden → num_experts
        gate_hidden = cfg_network['moe']['gate_mlp'].get('units', [128, 128])
        gate_activation = cfg_network['moe']['gate_mlp'].get('activation', 'relu')
        self.gate = build_mlp(gate_dim, gate_hidden, self.num_experts, gate_activation, self.dropout)
        self.gate = self.gate.to(self.device)

        # Initialize weights
        self._init_weights()
        
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def update_expert_num(self):
        if self.working_experts == self.num_experts:
            return
        self.working_experts += 4
        self.top_k += 1

    def forward(self, x, metrics=None):
        batch_size = x.size(0)

        # Gate scores
        gate_scores_logits_ = self.gate(x)  # [batch_size, num_experts]
        gate_scores_logits = gate_scores_logits_
        gate_scores = F.softmax(gate_scores_logits, dim=1)  # Softmax over experts for numerical stability

        # Top-k gate scores and indices
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=1)  # [batch_size, top_k]
        # if metrics is not None:
        #     print(top_k_indices)
        if metrics is not None:
            # print("***************")
            for i in range(self.num_experts):
                metrics['choose_expert_{}_rate'.format(i)] = (top_k_indices == i).sum().item() / batch_size
                # print("i: ", i, "rate: ", (top_k_indices == i).sum().item() / batch_size)

        # Pass inputs through all experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts])  # [num_experts, batch_size, output_dim]
        expert_outputs = expert_outputs.permute(1, 0, 2)  # [batch_size, num_experts, output_dim]

        # Advanced indexing for selecting top-k expert outputs
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.top_k).reshape(-1).to(x.device)
        selected_expert_outputs = expert_outputs[batch_indices, top_k_indices.reshape(-1)]  # [batch_size * top_k, output_dim]
        selected_expert_outputs = selected_expert_outputs.reshape(batch_size, self.top_k, self.output_dim)  # [batch_size, top_k, output_dim]

        # Scale the selected expert outputs by the corresponding gate scores
        scaled_expert_outputs = selected_expert_outputs * top_k_scores.unsqueeze(2)
        scaled_expert_outputs /= (top_k_scores.sum(dim=1, keepdim=True).unsqueeze(2) + 1e-9)  # Avoid division by zero

        # Sum the scaled expert outputs for the final output
        combined_output = scaled_expert_outputs.sum(dim=1)  # [batch_size, output_dim]

        if self.stmoe:
            acc_probs, acc_freq, acc_lsesq, acc_count\
                = stmoe_update_aux_stats(gate_scores_logits, gate_scores, top_k_indices)
            aux_loss = stmoe_loss(self.working_experts, acc_probs, acc_freq, acc_lsesq, acc_count)
        else:
            aux_loss = moe_auxiliary_loss(gate_scores, top_k_indices, self.lambda_balance, self.lambda_entropy)

        return combined_output, aux_loss, top_k_indices, top_k_scores

def moe_auxiliary_loss(gate_scores, top_k_indices, lambda_balance=5.0, lambda_entropy=0.1):
    batch_size, num_experts = gate_scores.size()

    # Load Balancing Loss
    one_hot = F.one_hot(top_k_indices, num_classes=num_experts).float()  # [batch_size, top_k, num_experts]
    expert_load = one_hot.sum(dim=[0, 1]) / (batch_size + 1e-9)  # Avoid division by zero; [num_experts]
    load_balancing_loss = expert_load.var()

    # Entropy Loss
    entropy = -(gate_scores * torch.log(gate_scores + 1e-9)).sum(dim=1).mean()  # Add epsilon to avoid log(0)

    # Combine the losses
    auxiliary_loss = lambda_balance * load_balancing_loss + lambda_entropy * entropy

    return auxiliary_loss

def stmoe_update_aux_stats(gate_scores_logits, gate_scores, top_k_indices):
    acc_count = 1
    _, num_experts = gate_scores_logits.size()
    acc_probs = gate_scores.sum(dim=0)
    acc_freq = F.one_hot(top_k_indices, num_classes=num_experts).float().sum(dim=[0, 1])
    acc_lsesq = torch.log(torch.exp(gate_scores_logits).sum(dim=-1)).pow(2).sum()

    return acc_probs, acc_freq, acc_lsesq, acc_count

def stmoe_loss(num_experts, acc_probs, acc_freq, acc_lsesq, acc_count):
    switch_loss = num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(acc_freq, p=1, dim=0)).sum()
    z_loss = acc_lsesq / acc_count
    loss = switch_loss + 0.1 * z_loss

    return loss
import torch.nn as nn
import torch
# from utils.torch_utils import  get_activation,check_cnnoutput
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
    

class VELO_NET(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',
                 decoder_hidden_dims = [512, 256, 128],
                 device='cpu'):
        super(VELO_NET, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history
        self.num_latent = num_latent 
        self.device = device

        # Build Encoder
    
        self.encoder = MLPHistoryEncoder(
            num_obs = num_obs,
            num_history=num_history,
            num_latent=num_latent * 2,
            activation=activation,
            adaptation_module_branch_hidden_dims=[512, 256],
        )
        # self.latent_mu = nn.Linear(num_latent * 4, num_latent)
        # self.latent_var = nn.Linear(num_latent * 4, num_latent)

        self.vel_mu = nn.Linear(num_latent * 2, 3)
        self.vel_var = nn.Linear(num_latent * 2, 3)

        self.height_mu = nn.Linear(num_latent * 2, 1)
        self.height_var = nn.Linear(num_latent * 2, 1)

        
    
    def encode(self, obs_history):
        encoded = self.encoder(obs_history)
        # latent_mu = self.latent_mu(encoded)
        # latent_var = self.latent_var(encoded)
        vel_mu = self.vel_mu(encoded)
        vel_var = self.vel_var(encoded)
        height_mu = self.height_mu(encoded)
        height_var = self.height_var(encoded)
        return [vel_mu, vel_var, height_mu, height_var]

   
    def forward(self, obs_history):
        vel_mu, vel_var, height_mu, height_var = self.encode(obs_history)
        
        vel = self.reparameterize(vel_mu, vel_var)
        height = self.reparameterize(height_mu, height_var)
        return [vel, height],[vel_mu, vel_var, height_mu, height_var]
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def loss_fn(self, obs_history, vel, height, velo_weight = 1.0, height_weight = 2.0):
        estimation, latent_params = self.forward(obs_history)
        v, h = estimation[0], estimation[1]
        vel_mu, vel_var, height_mu, height_var = latent_params 

        # Supervised loss
        v = torch.clamp(v.clone(), min=-1e2, max=1e2)
        vel = torch.clamp(vel.clone(), min=-1e2, max=1e2)
        # vel_loss = F.mse_loss(v, vel,reduction='none').mean(-1)
        # vel_loss = torch.nn.HuberLoss(delta=1.0)(v, vel)
        vel_loss = F.smooth_l1_loss(v, vel, reduction='none').mean(-1)

        h = torch.clamp(h.clone(), min=-1e2, max=1e2)
        height = torch.clamp(height.clone(), min=-1e2, max=1e2)
        height_loss = F.smooth_l1_loss(h, height, reduction='none').mean(-1)

        loss = torch.clamp(velo_weight*vel_loss+height_weight*height_loss, max=1e4)

        return loss
    

    def sample(self,obs_history):
        estimation, _ = self.forward(obs_history)
        return estimation
    def inference(self,obs_history):
        _, latent_params = self.forward(obs_history)
        vel_mu, vel_var, height_mu, height_var = latent_params
        return [vel_mu, height_mu]


# class TCNHistoryEncoder(nn.Module):
#     def __init__(self, 
#                  num_obs,
#                  num_history,
#                  num_latent,
#                  activation = 'elu',):
#         super(TCNHistoryEncoder, self).__init__()
#         self.num_obs = num_obs
#         self.num_history = num_history  
#         self.num_latent = num_latent    

#         activation_fn = get_activation(activation)
#         self.tsteps = tsteps = num_history
#         input_size = num_obs
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, 128),
#             activation_fn,
#             nn.Linear(128, 32),
#         )
#         if tsteps == 50:
#             self.conv_layers = nn.Sequential(
#                     nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
#                     nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
#                     nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
#             last_dim = 32 * 3
#         elif tsteps == 10:
#             self.conv_layers = nn.Sequential(
#                 nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
#                 nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
#                 nn.Flatten())
#             last_dim = 32 * 3
#         elif tsteps == 20:
#             self.conv_layers = nn.Sequential(
#                 nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
#                 nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
#                 nn.Flatten())
#             last_dim = 32 * 3
#         else:
#             self.conv_layers = nn.Sequential(
#                 nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
#                 nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
#                 nn.Flatten())
#             last_dim = check_cnnoutput(input_size = (32,self.tsteps), list_modules = [self.conv_layers])

#         self.output_layer = nn.Sequential(
#             nn.Linear(last_dim, self.num_latent),
#             activation_fn,
#         )


#     def forward(self, obs_history):
#         """
#         obs_history.shape = (bz, T , obs_dim)
#         """
#         bs = obs_history.shape[0]
#         T = self.tsteps
#         projection = self.encoder(obs_history) # (bz, T , 32) -> (bz, 32, T) bz, channel_dim, Temporal_dim
#         output = self.conv_layers(projection.permute(0, 2, 1)) # (bz, last_dim)
#         output = self.output_layer(output)
#         return output
    

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


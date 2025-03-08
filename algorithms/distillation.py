# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# seperate VAE net and velocity net

from multiprocessing.sharedctypes import Value
import sys, os

from torch.nn.utils.clip_grad import clip_grad_norm_

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time
import numpy as np
import copy
import torch
from torchvision import models, transforms
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, Any
from tensorboardX import SummaryWriter
import yaml

import dflex as df

import envs
import models.actor
import models.critic
from utils.common import *
import utils.torch_utils as tu
from utils.running_mean_std import RunningMeanStd
from utils.dataset import CriticDataset
from utils.time_report import TimeReport
from utils.average_meter import AverageMeter

from .vae_3 import VAE
from .velo_net import VELO_NET

import pdb
import wandb



class TeacherDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

class StudentPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StudentPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Distill:
    def __init__(self, cfg):
        env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        self.map_name = cfg["params"]["config"].get("map_name", 'gate_mid')

        seeding(cfg["params"]["general"]["seed"])
        self.env = env_fn(num_envs = cfg["params"]["config"]["num_actors"], \
                            device = cfg["params"]["general"]["device"], \
                            render = cfg["params"]["general"]["render"], \
                            seed = cfg["params"]["general"]["seed"], \
                            episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                            stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                            MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                            map_name = self.map_name,
                            no_grad = False)

        print('num_envs = ', self.env.num_envs)
        print('num_actions = ', self.env.num_actions)
        print('num_obs = ', self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_privilege_obs = self.env.num_privilege_obs
        self.num_actions = self.env.num_actions
        self.max_episode_length = self.env.episode_length
        self.device = cfg["params"]["general"]["device"]

        self.gamma = cfg['params']['config'].get('gamma', 0.99)
        
        self.critic_method = cfg['params']['config'].get('critic_method', 'one-step') # ['one-step', 'td-lambda']
        if self.critic_method == 'td-lambda':
            self.lam = cfg['params']['config'].get('lambda', 0.95)

        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.critic_lr = float(cfg['params']['config']['critic_learning_rate'])
        self.vae_lr = float(cfg['params']['config'].get('vae_learning_rate', 1e-4))
        self.vel_net_lr = float(cfg['params']['config'].get('vel_net_learning_rate', 1e-4))
        # lr schedule
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')
        self.init_actor_lr = self.actor_lr
        self.init_critic_lr = self.critic_lr
        self.init_vae_lr = self.vae_lr
        self.init_vel_net_lr = self.vel_net_lr
        # domain_randomization
        self.domain_randomization = cfg['params']['config'].get('domain_randomization', False)
        if self.domain_randomization:
            self.env.domain_randomization = True
        
        self.target_critic_alpha = cfg['params']['config'].get('target_critic_alpha', 0.4)
        self.curriculum = cfg['params']['config'].get('curriculum', False)
        self.obs_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            self.obs_rms = RunningMeanStd(shape = (self.num_obs), device = self.device)
            self.privilege_obs_rms = RunningMeanStd(shape = (self.num_privilege_obs), device = self.device)
            
        self.ret_rms = None
        if cfg['params']['config'].get('ret_rms', False):
            self.ret_rms = RunningMeanStd(shape = (), device = self.device)

        self.rew_norm = None
        if cfg['params']['config'].get('reward_norm', False):
            self.rew_norm = True
            rew_momentum = cfg['params']['config'].get('rew_momentum', 0.9)
            self.rew_mean = 0.0
            self.rew_var = 1.0

        self.multi_stage = cfg['params']['config'].get('multi_stage', False)
        self.stage_change_time = cfg['params']['config'].get('stage_change_time', 300)
        self.vel_net_early_stop = cfg['params']['config'].get('vel_net_early_stop', False)
        self.vel_net_stop_time = cfg['params']['config'].get('vel_net_stop_time', 300)

        # pretrain vel_net
        self.pretrain_vel_net = cfg['params']['config'].get('pretrain_vel_net', False)
        self.pretrain_epoch = cfg['params']['config'].get('pretrain_epoch', 30)
        self.pretrain_steps_num = cfg['params']['config'].get('pretrain_steps_num', 100)
        
        self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)

        self.critic_iterations = cfg['params']['config'].get('critic_iterations', 16)
        self.num_batch = cfg['params']['config'].get('num_batch', 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg['params']['config'].get('name', "Ant")

        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm = cfg["params"]["config"]["grad_norm"]

        

        # create actor critic network
        self.actor_name = cfg["params"]["network"].get("actor", 'ActorStochasticMLP') # choices: ['ActorDeterministicMLP', 'ActorStochasticMLP']
        self.critic_name = cfg["params"]["network"].get("critic", 'CriticMLP')
        actor_fn = getattr(models.actor, self.actor_name)
        self.actor = actor_fn(self.num_obs, self.num_actions, cfg['params']['network'], device = self.device)
        critic_fn = getattr(models.critic, self.critic_name)
        self.critic = critic_fn(self.num_privilege_obs, cfg['params']['network'], device = self.device)
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.target_critic = copy.deepcopy(self.critic)

        # creat VAE
        self.num_history = self.env.num_history
        self.num_latent = self.env.num_latent
        self.vae = VAE(self.num_obs, self.num_history, self.num_latent, activation = 'elu', decoder_hidden_dims = [512, 256, 128]).to(self.device)
        self.kl_weight = cfg["params"]["vae"].get("kl_weight", 1.0)
        self.velo_weight = cfg["params"]["vae"].get("velo_weight", 1.0)
        # create VELO_NET
        self.vel_net = VELO_NET(self.num_obs, self.num_history, self.num_latent, activation = 'elu', decoder_hidden_dims = [512, 256, 128]).to(self.device)
    

        
       
        # replay buffer
        self.obs_buf = torch.zeros((self.steps_num, self.num_envs, self.num_obs), dtype = torch.float32, device = self.device)
        self.privilege_obs_buf = torch.zeros((self.steps_num, self.num_envs, self.num_privilege_obs), dtype = torch.float32, device = self.device)
        self.obs_hist_buf = torch.zeros((self.num_envs, self.num_obs*self.num_history), dtype = torch.float32, device = self.device)
        self.rew_buf = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.done_mask = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.next_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.target_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.ret = torch.zeros((self.num_envs), dtype = torch.float32, device = self.device)

        
        if cfg['params']['general']['checkpoint'] != 'Base':
            ckpt_path = cfg['params']['general']['checkpoint']
            ckpt_dir = os.path.dirname(ckpt_path)
            self.load(cfg['params']['general']['checkpoint'])
            print_info(f'actor critic networks recovered from {ckpt_path}')
           
        
        # for kl divergence computing
        self.old_mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.old_sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int, device = self.device)
        # self.episode_length = torch.zeros(self.num_envs, dtype = int)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf
        self.vae_loss = np.inf
        self.vel_net_loss = np.inf
        self.mean_recons_loss = np.inf
        self.mean_vel_loss = np.inf
        self.mean_kld_loss = np.inf
        
        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        # timer
        self.time_report = TimeReport()

        # grad recore
        self.max_grad = 0
        self.mean_grad = 0


    def distill_policy(self, teacher_dir, teacher_name):
      
        # Load the teacher policy
        teacher_policy_path = f"{teacher_dir}/{teacher_name}"
        print(f'loading teacher policy: {teacher_policy_path}')
        # ckpt = torch.load(teacher_policy_path, map_location=device)
        ckpt = torch.load(teacher_policy_path)
        teacher_policy = ckpt[0].to(self.device)
        teacher_policy.eval()

        # Simulated environment data (replace this with actual environment interaction)
        def generate_teacher_data(teacher, num_samples=1000):
            observations = []
            actions = []
            for _ in range(num_samples):
                # Simulate observation (random for demonstration)
                obs = torch.randn([1, 62], device=self.device)  # Original teacher observation size
                obs_without_lin_vel = torch.cat([obs[:, :19], obs[:, 22:]], dim=-1)  # Remove lin_vel (index 1)
                observations.append(obs_without_lin_vel)

                # Get teacher action
                with torch.no_grad():
                    action = teacher(obs.unsqueeze(0))
                actions.append(action.squeeze(0))

            return torch.stack(observations), torch.stack(actions)

        teacher_obs, teacher_actions = generate_teacher_data(teacher_policy)

        # Create dataset and dataloader
        dataset = TeacherDataset(teacher_obs, teacher_actions)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Initialize student policy
        student_policy = StudentPolicy(input_dim=59, output_dim=teacher_actions.size(-1)).to(self.device)  # 61 after removing lin_vel
        optimizer = optim.Adam(student_policy.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Distillation training loop
        epochs = 50
        for epoch in range(epochs):
            student_policy.train()
            epoch_loss = 0.0
            for batch_obs, batch_actions in dataloader:
                batch_obs, batch_actions = batch_obs.to(self.device), batch_actions.to(self.device)
                optimizer.zero_grad()
                student_output = student_policy(batch_obs)
                loss = criterion(student_output, batch_actions)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        # Save the trained student policy
        torch.save(student_policy.state_dict(), os.path.join(teacher_dir, 'student_policy.pt'))
        print(f"Student policy saved to {teacher_dir}")


    
    
import sys, os
os.environ["WANDB_MODE"] = "disabled"
from torch.nn.utils.clip_grad import clip_grad_norm_
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time
import numpy as np
import copy
import torch
from torchvision import models, transforms
import torch.nn as nn
import math
import yaml
import envs
import models.actor
import models.critic
from utils.common import *
import utils.torch_utils as tu
from utils.running_mean_std import RunningMeanStd
from utils.dataset import CriticDataset
from utils.time_report import TimeReport
from utils.average_meter import AverageMeter
from models.vae import VAE
import wandb
torch.autograd.set_detect_anomaly(True)

class BatchLowPassFilter:
    def __init__(self, alpha=0.1, batch_size=1):
        self.alpha = alpha
        self.last_values = None
    def filter(self, new_values):
        if self.last_values is None:
            self.last_values = new_values.clone().detach()
        else:
            self.last_values = self.alpha * new_values + (1 - self.alpha) * self.last_values.clone().detach()
        return self.last_values

class GradNav:
    def __init__(self, cfg):
        env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
        self.map_name = cfg["params"]["config"].get("map_name", 'gate_mid')

        seeding(cfg["params"]["general"]["seed"])
        env_hyper = cfg["params"].get("env_hyper", None)
        self.env = env_fn(num_envs = cfg["params"]["config"]["num_actors"], \
                            device = cfg["params"]["general"]["device"], \
                            render = cfg["params"]["general"]["render"], \
                            seed = cfg["params"]["general"]["seed"], \
                            episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                            stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                            MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                            map_name = self.map_name,
                            env_hyper = env_hyper,
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
        # lr schedule
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')
        self.init_actor_lr = self.actor_lr
        self.init_critic_lr = self.critic_lr
        self.init_vae_lr = self.vae_lr
        # domain_randomization
        self.domain_randomization = cfg['params']['config'].get('domain_randomization', False)
        if self.domain_randomization:
            self.env.domain_randomization = True
        
        self.target_critic_alpha = cfg['params']['config'].get('target_critic_alpha', 0.4)
        self.obs_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            self.obs_rms = RunningMeanStd(shape = (self.num_obs), device = self.device)
            self.privilege_obs_rms = RunningMeanStd(shape = (self.num_privilege_obs), device = self.device)
            
        self.ret_rms = None
        if cfg['params']['config'].get('ret_rms', False):
            self.ret_rms = RunningMeanStd(shape = (), device = self.device)

        self.change_gate_count = 0
        self.multi_gate = cfg['params']['config'].get('multi_gate', False)
        self.gate_change_time = cfg['params']['config'].get('gate_change_time', 150)

        self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)
        self.critic_iterations = cfg['params']['config'].get('critic_iterations', 16)
        self.num_batch = cfg['params']['config'].get('num_batch', 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg['params']['config'].get('name', "Ant")

        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm = cfg["params"]["config"]["grad_norm"]

        # LPF
        self.train_LPF = cfg['params']['config'].get('LPF_train', False)
        self.train_LPF_val = cfg['params']['config'].get('LPF_val', 0.5)
        if self.train_LPF:
            self.train_r_filter = BatchLowPassFilter(alpha=self.train_LPF_val, batch_size=self.num_envs)
            self.train_p_filter = BatchLowPassFilter(alpha=self.train_LPF_val, batch_size=self.num_envs)
            self.train_y_filter = BatchLowPassFilter(alpha=self.train_LPF_val, batch_size=self.num_envs)
            self.train_thrust_filter = BatchLowPassFilter(alpha=self.train_LPF_val, batch_size=self.num_envs)

        self.eval_LPF = cfg['params']['config']['player'].get('LPF_eval', False)
        self.eval_LPF_val = cfg['params']['config']['player'].get('LPF_val', 0.5)
        if self.eval_LPF:
            self.eval_r_filter = BatchLowPassFilter(alpha=self.eval_LPF_val, batch_size=self.num_envs)
            self.eval_p_filter = BatchLowPassFilter(alpha=self.eval_LPF_val, batch_size=self.num_envs)
            self.eval_y_filter = BatchLowPassFilter(alpha=self.eval_LPF_val, batch_size=self.num_envs)
            self.eval_thrust_filter = BatchLowPassFilter(alpha=self.eval_LPF_val, batch_size=self.num_envs)


        
        if cfg['params']['general']['train']:
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok = True)
            # save config
            save_cfg = copy.deepcopy(cfg)
            if 'general' in save_cfg['params']:
                deleted_keys = []
                for key in save_cfg['params']['general'].keys():
                    if key in save_cfg['params']['config']:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg['params']['general'][key]

            yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
            # save interval
            self.save_interval = cfg["params"]["config"].get("save_interval", 500)
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (cfg['params']['config']['player'].get('determenistic', False) or cfg['params']['config']['player'].get('deterministic', False))
            self.steps_num = self.env.episode_length

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
        self.kl_weight = cfg["params"]["vae"].get("kl_weight", 1.0)
        self.vae_encoder_dim = cfg["params"]["vae"].get("encoder_units", [256, 256, 256])
        self.vae_decoder_dim = cfg["params"]["vae"].get("decoder_units", [32, 64, 128, 256])
        self.vae = VAE(self.num_obs, self.num_history, self.num_latent, kld_weight=self.kl_weight, activation='elu', decoder_hidden_dims=self.vae_decoder_dim, encoder_hidden_dims=self.vae_encoder_dim).to(self.device)


        if cfg['params']['general']['train']:
            self.save('init_policy')

        # initialize optimizer
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), betas = cfg['params']['config']['betas'], lr = self.actor_lr, weight_decay=5e-3)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), betas = cfg['params']['config']['betas'], lr = self.critic_lr, weight_decay=5e-3)
        self.vae_optimizer = torch.optim.AdamW(self.vae.parameters(), betas = cfg['params']['config']['betas'], lr = self.vae_lr, weight_decay=1e-2)

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
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf
        self.vae_loss = np.inf
        self.mean_recons_loss = np.inf
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

        
    def compute_actor_loss(self, deterministic = False):
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)

        actor_loss = torch.tensor(0., dtype = torch.float32, device = self.device)
        vae_loss = torch.tensor(0., dtype = torch.float32, device = self.device)
        mean_recons_loss = 0
        mean_kld_loss = 0

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)
                privilege_obs_rms = copy.deepcopy(self.privilege_obs_rms)
            
            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # initialize trajectory to cut off gradients between episodes.
        obs, privilege_obs = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                self.obs_rms.update(obs)
                self.privilege_obs_rms.update(privilege_obs)
            # normalize the current obs
            obs = obs_rms.normalize(obs)
            privilege_obs = privilege_obs_rms.normalize(privilege_obs)
        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad():
                # # TODO: ugly implementation
                obs = torch.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
                privilege_obs = torch.nan_to_num(privilege_obs, nan=0.0, posinf=1e3, neginf=-1e3)
                self.obs_buf[i] = obs.clone()
                self.privilege_obs_buf[i] = privilege_obs.clone()

            # LPF
            if self.train_LPF:
                # Add LPF
                actions = self.actor(obs, deterministic=deterministic)
                actions = torch.tanh(actions)
                # Apply the LPF (construct a new tensor for filtered actions)
                filtered_actions = torch.stack([
                    self.train_r_filter.filter(actions[:, 0]),
                    self.train_p_filter.filter(actions[:, 1]),
                    self.train_y_filter.filter(actions[:, 2]),
                    self.train_thrust_filter.filter(actions[:, 3])
                ], dim=1)
                # Continue with the rest of your pipeline
                vae_output, _ = self.vae.forward(self.obs_hist_buf)
                obs, privilege_obs, history_obs, vel_obs, rew, done, extra_info = self.env.step(filtered_actions, vae_output)
            else:
                # No LPF
                actions = self.actor(obs, deterministic = deterministic)
                vae_output, _ = self.vae.forward(self.obs_hist_buf)
                obs, privilege_obs, history_obs, vel_obs, rew, done, extra_info = self.env.step(torch.tanh(actions), vae_output)

            self.obs_hist_buf = history_obs

            # update VAE
            recons_loss, kld_loss, vae_loss_grad = self.compute_vae_loss(obs, vel_obs, history_obs, done)
            mean_kld_loss = mean_kld_loss + kld_loss
            mean_recons_loss = mean_recons_loss + recons_loss
            vae_loss = vae_loss + vae_loss_grad
            
            with torch.no_grad():
                raw_rew = rew.clone()
            
            # scale the reward
            rew = rew * self.rew_scale
            
            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                    self.privilege_obs_rms.update(privilege_obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)
                privilege_obs = privilege_obs_rms.normalize(privilege_obs)

            if self.ret_rms is not None:
                # update ret rms
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)
                rew = rew / torch.sqrt(ret_var + 1e-6)


            self.episode_length += 1
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)
            next_values[i + 1] = self.target_critic(privilege_obs).squeeze(-1)

            for id in done_env_ids:
                if torch.isnan(extra_info['obs_before_reset'][id]).sum() > 0 \
                    or torch.isinf(extra_info['obs_before_reset'][id]).sum() > 0 \
                    or (torch.abs(extra_info['obs_before_reset'][id]) > 1e6).sum() > 0: 
                    next_values[i + 1, id] = 0.
                elif self.episode_length[id] < self.max_episode_length: # early termination
                    next_values[i + 1, id] = 0.
                else: # otherwise, use terminal value critic to estimate the long-term performance
                    if self.obs_rms is not None:
                        real_obs = obs_rms.normalize(extra_info['obs_before_reset'][id])
                        real_privilege_obs = privilege_obs_rms.normalize(extra_info['privilege_obs_before_reset'][id])
                    else:
                        real_obs = extra_info['obs_before_reset'][id]
                        real_privilege_obs = extra_info['privilege_obs_before_reset'][id]
                    next_values[i + 1, id] = self.target_critic(real_privilege_obs).squeeze(-1)
            
            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                # raise ValueError
            
            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
            else:
                actor_loss = actor_loss + (- rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()
        
            gamma = gamma * self.gamma
            gamma[done_env_ids] = 1.
            rew_acc[i + 1, done_env_ids] = 0.

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    for done_env_id in done_env_ids:
                        if (self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6):
                            print('ep loss exploded')

                        self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.
                        self.episode_discounted_loss[done_env_id] = 0.
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.

        actor_loss /= self.steps_num * self.num_envs
        vae_loss /= self.steps_num 
        mean_recons_loss /= self.steps_num
        mean_kld_loss /= self.steps_num

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)
            
        self.actor_loss = actor_loss.detach().cpu().item()
        self.mean_recons_loss = mean_recons_loss
        self.mean_kld_loss = mean_kld_loss
        self.vae_loss = mean_recons_loss + self.kl_weight * mean_kld_loss
        self.step_count += self.steps_num * self.num_envs


        return actor_loss, vae_loss
    
    def vae_update(self, vae_loss):
        self.time_report.start_timer("VAE training")
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        nn.utils.clip_grad_norm_(self.vae.parameters(), self.grad_norm)
        self.vae_optimizer.step()
        self.time_report.end_timer("VAE training")

    def compute_vae_loss(self, obs, vel_obs, history_obs, done):
        self.time_report.start_timer("VAE training")
        vae_loss_dict = self.vae.loss_fn(history_obs.clone().detach(), obs.clone().detach())
        valid = (done == 0).squeeze()
        vae_loss = torch.mean(vae_loss_dict['loss'][valid])
        recons_loss = torch.mean(vae_loss_dict['recons_loss'][valid])
        kld_loss = torch.mean(vae_loss_dict['kld_loss'][valid]) 
        self.time_report.end_timer("VAE training")
        return recons_loss.item(), kld_loss.item(), vae_loss
    
    
    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic = False):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        episode_length = torch.zeros(self.num_envs, dtype = int)
        episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        obs = self.env.reset()
        games_cnt = 0

        while games_cnt < num_games:
            raw_obs = obs.clone()
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            if self.eval_LPF:
                # Add LPF
                actions = self.actor(obs, deterministic=deterministic)
                actions = torch.tanh(actions)
                # Apply the LPF
                actions[:, 0] = self.eval_r_filter.filter(actions[:, 0])
                actions[:, 1] = self.eval_p_filter.filter(actions[:, 1])
                actions[:, 2] = self.eval_y_filter.filter(actions[:, 2])
                actions[:, 3] = self.eval_thrust_filter.filter(actions[:, 3])
                vae_output, _ = self.vae.forward(self.obs_hist_buf)
                obs, privilege_obs, history_obs, vel_obs, rew, done, extra_info = self.env.step(actions, vae_output)
            else:
                # No LPF
                actions = self.actor(obs, deterministic = deterministic)
                vae_output, _ = self.vae.forward(self.obs_hist_buf)
                obs, privilege_obs, history_obs, vel_obs, rew, done, extra_info = self.env.step(torch.tanh(actions), vae_output)
            
            self.obs_hist_buf = history_obs
            episode_length += 1
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print('loss = {:.2f}, len = {}'.format(episode_loss[done_env_id].item(), episode_length[done_env_id]))
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(episode_discounted_loss[done_env_id].item())
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.
                    episode_discounted_loss[done_env_id] = 0.
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.
                    games_cnt += 1
        
        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_loss = np.mean(np.array(episode_loss_his))
        mean_policy_discounted_loss = np.mean(np.array(episode_discounted_loss_his))
 
        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            Bi = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            lam = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (1. - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError
            
    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
        target_values = batch_sample['target_values']
        critic_loss = ((predicted_values - target_values) ** 2).mean()

        return critic_loss

    def initialize_env(self):
        self.env.clear_grad()
        self.env.reset()

    @torch.no_grad()
    def run(self, num_games):
        mean_policy_loss, mean_policy_discounted_loss, mean_episode_length = self.evaluate_policy(num_games = num_games, deterministic = not self.stochastic_evaluation)
        print_info('mean episode loss = {}, mean discounted loss = {}, mean episode length = {}'.format(mean_policy_loss, mean_policy_discounted_loss, mean_episode_length))


    def train(self):
        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")
        self.time_report.add_timer("VAE training")

        self.time_report.start_timer("algorithm")

        # initializations
        self.initialize_env()
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        
        def actor_closure():
            self.actor_optimizer.zero_grad()

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            actor_loss, vae_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                self.max_grad, self.mean_grad = tu.calculate_max_mean_gradient(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters()) 
                
                # sanity check
                if torch.isnan(self.grad_norm_before_clip) or self.grad_norm_before_clip > 1000000.:
                    print('NaN gradient')
                    raise ValueError

            self.time_report.end_timer("compute actor loss")
            self.vae_update(vae_loss)

            return actor_loss

        # setup wandb
        self.hyper_parameter = self.env.hyper_parameter
        train_parameter = {
            "learning_rate": self.actor_lr,
            "architecture": "shac",
            "epochs": self.max_epochs,
            "episodes": self.max_episode_length,
            "batchsize": self.batch_size,
            "num_players": self.env.num_envs,
            }
        self.hyper_parameter.update(train_parameter)

        self.wandb_start = False
        try:
            wandb.init(
                project=f'{self.env.agent_name}-{self.map_name}',
                config=self.hyper_parameter
            )
            self.wandb_start = True
        except Exception as e:
            print_info(f'wandb init failed: {e}')
            self.wandb_start = False
            

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()
            # Multi gate training process
            if self.multi_gate:
                if (epoch+1) % self.gate_change_time == 0:
                    self.change_gate_count += 1
                    curr_map_ptr = self.change_gate_count % self.env.multi_map_num
                    next_map_name = self.env.multi_map[curr_map_ptr]
                    self.initialize_env()
                    self.env.change_map(next_map_name)
                    self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), betas = [0.7, 0.95], lr = self.actor_lr, weight_decay=5e-3)
                    print_info(f'map has been changed to {next_map_name}, actor lr reset to {self.actor_lr}')


            # learning rate schedule
            if self.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
                vae_lr = (1e-5 - self.vae_lr) * float(epoch / self.max_epochs) + self.vae_lr
                for param_group in self.vae_optimizer.param_groups:
                    param_group['lr'] = vae_lr
            # learning rate schedule
            elif self.lr_schedule == 'cosine':
                # Calculate the cosine schedule
                actor_lr = 1e-5 + (self.actor_lr - 1e-5) *  0.5 * (1 + math.cos(math.pi * epoch / self.max_epochs))
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = 1e-5 + (self.critic_lr - 1e-5)  * 0.5 * (1 + math.cos(math.pi * epoch / self.max_epochs))
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
                vae_lr = 1e-5 + (self.vae_lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * epoch / self.max_epochs))
                for param_group in self.vae_optimizer.param_groups:
                    param_group['lr'] = vae_lr
            else:
                lr = self.actor_lr


            # train actor
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure).detach().item()
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.privilege_obs_buf, self.target_values, drop_last = False)
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.
            for j in range(self.critic_iterations):
                total_critic_loss = 0.
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()
                    
                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()
                    total_critic_loss += training_critic_loss
                    batch_cnt += 1
                
                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, self.value_loss), end='\r')

            self.time_report.end_timer("critic training")

            self.iter_count += 1
            
            time_end_epoch = time.time()

            # logging
            time_elapse = time.time() - self.start_time
            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            print('iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, vae loss {:.2f}, ep len {:.1f}, fps total {:.2f}, value loss {:.2f}, grad norm before clip {:.2f}, grad norm after clip {:.2f}'.format(\
                    self.iter_count, mean_policy_loss, mean_policy_discounted_loss, self.vae_loss, mean_episode_length, self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch), self.value_loss, self.grad_norm_before_clip, self.grad_norm_after_clip))

            # self.writer.flush()
        
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save("policy_iter{}_reward{:.3f}".format(self.iter_count, -mean_policy_loss))
                

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)
            
            
            # update wandb
            wandb_record_loss = mean_policy_loss
            if self.wandb_start:
                wandb.log({"actor loss": wandb_record_loss, 
                        "VAE_loss": self.vae_loss,
                        "recons_loss": self.mean_recons_loss,
                        "kld_loss": self.mean_kld_loss,
                        "episode_length": mean_episode_length,
                        "max_grad":self.max_grad,
                            "mean_grad": self.mean_grad,
                            "learning rate": actor_lr
                        })

        self.time_report.end_timer("algorithm")
        self.time_report.report()
        self.env.time_report.report()
        
        self.save('final_policy')
        self.episode_loss_his = np.array(self.episode_loss_his)
        self.episode_discounted_loss_his = np.array(self.episode_discounted_loss_his)
        self.episode_length_his = np.array(self.episode_length_his)
        
        # evaluate the final policy's performance
        self.run(self.num_envs)
    
    def play(self, cfg):
        ckpt_path = cfg['params']['general']['checkpoint']
        ckpt_dir = os.path.dirname(ckpt_path)
        self.load(cfg['params']['general']['checkpoint'])
        self.run(cfg['params']['config']['player']['games_num'])

    def save(self, filename = None):
        if filename is None:
            filename = 'best_policy'
        torch.save([self.actor, self.critic, self.target_critic, self.obs_rms, self.vae, self.env.visual_net], os.path.join(self.log_dir, "{}.pt".format(filename)))

    def load(self, path):
        print(path)
        checkpoint = torch.load(path)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.target_critic = checkpoint[2].to(self.device)
        self.obs_rms = checkpoint[3].to(self.device)
        self.vae = checkpoint[4].to(self.device)
        self.env.visual_net = checkpoint[5].to(self.device)
        print_info(f"all nets have been loaded")

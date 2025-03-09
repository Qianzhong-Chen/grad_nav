# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from envs.dflex_env import DFlexEnv
import math
import torch
torch.autograd.set_detect_anomaly(True)

from torchvision import models, transforms
from torchvision.transforms import Resize
from torchvision.io import write_video
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from rich.console import Console
from PIL import Image
from envs.assets.quadrotor_dynamics_advanced import QuadrotorSimulator
# from envs.assets.visiual_buffer import VisualBuffer

import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

# try:
#     from pxr import Usd
# except ModuleNotFoundError:
#     print("No pxr package")

# from utils import load_utils as lu
from utils import torch_utils as tu
from utils.common import *
from utils.gs_local import GS, get_gs
from utils.rotation import dynamic_to_nerf_quaternion_batch, pose_transfer_ns, pose_transfer_ns_batched, quaternion_to_euler, quaternion_yaw_forward
from utils.model_util import quat_from_axis_angle, transform, quat_from_axis_angle_batch
from utils.model import ModelBuilder
from utils.point_cloud_util import ObstacleDistanceCalculator
from utils.traj_planner_global import TrajectoryPlanner
from utils.hist_obs_buffer import ObsHistBuffer
from utils.time_report import TimeReport
import time
import matplotlib.pyplot as plt
# squeezenet + trainable linear layer
import torchvision.models as models

SINGLE_VISUAL_INPUT_SIZE = 24
HISTORY_BUFFER_NUM = 5
LATENT_VECT_NUM = 16

class VisualPerceptionNet(nn.Module):
    def __init__(self, input_channels=4):
        super(VisualPerceptionNet, self).__init__()
        # Set up the SqueezeNet backbone
        self.squeezenet = models.squeezenet1_0(pretrained=True)
        
        # Modify the first convolution layer to accept more channels (e.g., 4 for RGB + Depth)
        self.squeezenet.features[0] = nn.Conv2d(input_channels, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Freeze SqueezeNet parameters so they are not updated during training
        for param in self.squeezenet.parameters():
            param.requires_grad = False

        # Add an adaptive average pooling to get the output to [batch_size, 512, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a 1D convolution layer to downsample from 512 to 32
        # self.conv1d = nn.Conv1d(512, SINGLE_VISUAL_INPUT_SIZE, kernel_size=1)
        self.fc = nn.Linear(512, SINGLE_VISUAL_INPUT_SIZE)

    def forward(self, x):
        # Freeze the SqueezeNet layers but keep the 1D convolution layer trainable
        with torch.no_grad():
            x = self.squeezenet.features(x)  # SqueezeNet backbone is frozen
            x = self.pool(x)  # Adaptive pooling to reduce spatial dimensions to 1x1
        x = torch.flatten(x, 1)  # Flatten to shape [batch_size, 512]
        
        # Conv
        # x = x.unsqueeze(-1)  # Add a channel dimension for 1D convolution
        # x = self.conv1d(x).squeeze(-1)  # Apply 1D convolution and remove the extra dimension
        
        # FC layer
        x = self.fc(x)

        return x


class DronePaperTestEnv(DFlexEnv):

    def __init__(self, 
                 render=False, 
                 device='cuda:0', 
                 num_envs=4096, 
                 seed=0, 
                 episode_length=1000, 
                 no_grad=True, 
                 stochastic_init=False, 
                 MM_caching_frequency = 1, 
                 early_termination = True, 
                 map_name='gate_mid'
                 ):
       
        self.num_privilege_obs = 30 + SINGLE_VISUAL_INPUT_SIZE + LATENT_VECT_NUM
        num_obs = 22 + SINGLE_VISUAL_INPUT_SIZE + LATENT_VECT_NUM # correspond with self.obs_buf
        num_act = 4
        self.agent_name = 'drone_paper_test'
        self.num_history = HISTORY_BUFFER_NUM
        self.num_latent = LATENT_VECT_NUM
    
        print('----------------------------', device)
        super(DronePaperTestEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.device = device
        self.height_target = 1.5
        self.time_report = TimeReport()
        self.time_report.add_timer("dynamic simulation")
        self.time_report.add_timer("3D GS inference")
        self.time_report.add_timer("point cloud collision check")

        # setup map
        maps = {
            "gate_left":"sv_917_3_left_nerfstudio",
            "gate_right":"sv_917_3_right_nerfstudio",
            "simple_hover":"sv_917_3_right_nerfstudio",
            "gate_mid":"sv_1007_gate_mid",
            "clutter":"sv_712_nerfstudio",
            "backroom":"sv_1018_2",
            "flightroom":"sv_1018_3",
        }
        self.map_name = map_name

        # SysID value
        self.min_mass = 1.0
        self.mass_range = 0.2
        self.min_thrust = 24.0
        self.thrust_range = 2.0
        self.obs_noise_level = 0.1

        self.init_sim()
        self.episode_length = episode_length

        # GS model
        gs_dir = Path("assets/gs_data")
        resolution_quality = 0.4
        self.gs = get_gs(self.map_name, gs_dir, resolution_quality)
        

        # navigation parameters

        
        if self.map_name == 'gate_mid':
            target = (13.0, -2.0, 1.5) # in map absolute dimension, start with 0
            target_xy = (13.0, -2.0)
        if self.map_name == 'gate_left':
            target = (13.0, -2.0, 1.2) # in map absolute dimension, start with 0
            target_xy = (13.0, -2.0)
        if self.map_name == 'gate_right':
            target = (13.0, -2.0, 1.5) # in map absolute dimension, start with 0
            target_xy = (13.0, -2.0)
        if self.map_name == 'simple_hover':
            target = (3.0, 0.0, 1.2) # in map absolute dimension, start with 0
            target_xy = (3.0, 0.0)
        if self.map_name == 'clutter':
            target = (13.0, -2.0, 1.2) # in map absolute dimension, start with 0
            target_xy = (13.0, -2.0)
        self.target = tu.to_torch(list(target), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # navigation target
        self.target_xy = tu.to_torch(list(target_xy), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # navigation target
        # self.target_list = self.target.clone()
        self.gs_origin_offset = torch.tensor([[-6.0, -0., -0.2]] * self.num_envs, device=self.device) # (x,y,z); gs dimension for visual info
        self.point_could_origin_offset = torch.tensor([[-6.0, -0., -0.2]] * self.num_envs, device=self.device) # (x,y,z); point cloud dimension for obstacle distance & traj plan

        # setup point cloud
        point_cloud_dir = Path("assets/point_cloud")
        self.point_cloud = ObstacleDistanceCalculator(ply_file=os.path.join(point_cloud_dir,f'{maps[self.map_name]}.ply'))
        self.traj_planner = TrajectoryPlanner(ply_file=os.path.join(point_cloud_dir,f'{maps[self.map_name]}.ply'), 
                                            safety_distance=0.15, 
                                            batch_size=1, 
                                            wp_distance=2.0,
                                            verbose=False)
        
        # # For ablation test: generate ref_traj
        if self.map_name == 'gate_mid':
            self.reward_wp = torch.tensor([[5.8, -0.1, 1.4], 
                                        [9.7, 1.5, 0.6], 
                                        [11.8, 0.0, 1.2],
                                        [13.0, -2.0, 1.2]], device=self.device) # (x,y,z)
            
            
            # plane the global ref trajectory
            traj_wp = self.reward_wp + self.point_could_origin_offset[0].repeat((self.reward_wp.shape[0],1))
            traj_start = torch.tensor([[-6.0, 0, 1.2]], device=self.device)
            traj_dest = torch.tensor([[7.0, -2.0, 1.2]], device=self.device)
            self.ref_traj = self.traj_planner.plan_trajectories(traj_start, traj_dest, [traj_wp])
            self.ref_traj = torch.tensor(self.ref_traj[0], device = self.device)

        

         # generate ref_traj
        if self.map_name == 'gate_left':
            self.reward_wp = torch.tensor([[5.8, 1.2, 1.4], 
                                        [9.7, 1.2, 0.6], 
                                        [11.8, 0.0, 1.2],
                                        [13.0, -2.0, 1.2]], device=self.device) # (x,y,z)
            
            
            # plane the global ref trajectory
            traj_wp = self.reward_wp + self.point_could_origin_offset[0].repeat((self.reward_wp.shape[0],1))
            traj_start = torch.tensor([[-6.0, 0, 1.2]], device=self.device)
            traj_dest = torch.tensor([[7.0, -2.0, 1.2]], device=self.device)
            self.ref_traj = self.traj_planner.plan_trajectories(traj_start, traj_dest, [traj_wp])
            self.ref_traj = torch.tensor(self.ref_traj[0], device = self.device)

         # generate ref_traj
        if self.map_name == 'gate_right':
            self.reward_wp = torch.tensor([
                                        [3.0, -1.3, 1.5],
                                        [6.2, -1.0, 1.6],
                                        [7.8, 0.3, 1.3],
                                        [9.7, 1.3, 0.9],
                                        [11.8, 0, 1.5],
                                        [13.0, -2., 1.5]
                                        ], device=self.device) # (x,y,z)


            # plane the global ref trajectory
            traj_wp = self.reward_wp + self.point_could_origin_offset[0].repeat((self.reward_wp.shape[0],1))
            traj_start = torch.tensor([[-6.0, 0, 1.4]], device=self.device)
            traj_dest = torch.tensor([[7.0, -2.0, 1.4]], device=self.device)
            self.ref_traj = self.traj_planner.plan_trajectories(traj_start, traj_dest, [traj_wp])
            self.ref_traj = torch.tensor(self.ref_traj[0], device = self.device)

        if self.map_name == 'simple_hover':
            self.reward_wp = torch.tensor([[3.0, 0.0, 1.2], 
                                        ], device=self.device) # (x,y,z)
            
            
            # plane the global ref trajectory
            traj_wp = self.reward_wp + self.point_could_origin_offset[0].repeat((self.reward_wp.shape[0],1))
            traj_start = torch.tensor([[-6.0, 0, 1.2]], device=self.device)
            traj_dest = torch.tensor([[-3.0, 0.0, 1.2]], device=self.device)
            self.ref_traj = self.traj_planner.plan_trajectories(traj_start, traj_dest, [traj_wp])
            self.ref_traj = torch.tensor(self.ref_traj[0], device = self.device)
        
        if self.map_name == 'clutter':
            self.reward_wp = torch.tensor([
                              [1.1, -0.6, 1.4],
                              [6.1, -1.7, 0.6],
                              [9.4, 1.5, 1.2],
                              [12.0, -2.3, 1.2]], device=self.device)
            # plane the global ref trajectory
            traj_wp = self.reward_wp + self.point_could_origin_offset[0].repeat((self.reward_wp.shape[0],1))
            traj_start = torch.tensor([[-6.0, 1.0, 1.2]], device=self.device)
            traj_dest = torch.tensor([[6.0, -2.3, 1.2]], device=self.device)
            self.ref_traj = self.traj_planner.plan_trajectories(traj_start, traj_dest, [traj_wp])
            self.ref_traj = torch.tensor(self.ref_traj[0], device = self.device)
        
        
        
        self.reward_wp_list = [self.reward_wp + self.point_could_origin_offset[0].repeat((self.reward_wp.shape[0],1))] * self.num_envs
        self.reward_wp_record = []
        for i in range(self.reward_wp.shape[0]):
            self.reward_wp_record.append(torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        
        
        
        # Other parameters
        if self.map_name == 'gate_mid':
            # ## Original way
            # self.termination_distance = 1000
            # self.action_strength = 1.0 # only act on body rate
            # self.joint_vel_obs_scaling = 0.1
            self.up_strength = 0.25
            self.heading_strength = 0.25 # reward yaw motion align with lin_vel
            self.lin_strength = -1.5
            self.lin_vel_rate_penalty = 0.
            self.yaw_strength = 0.0
            self.action_penalty = -1.0
            self.action_change_penalty = -1.0
            self.smooth_penalty = -1.0
            self.smooth_penalty = -1.0
            # self.ang_vel_penalty = -0.4
            # self.ang_acc_penalty = -0.005
            self.survive_reward = 8.0
            self.pose_penalty = -0.5
            self.pose_penalty = -0.5
            self.height_penalty = -2.0
            self.height_change_penalty = -0.5
            self.height_change_penalty = -0.5
            self.jitter_penalty = -0.1
            self.out_map_penalty = -1.5
            self.map_center_penalty = -0.05
            # trajectory
            self.waypoint_strength = 2.0
            self.target_factor = -2.
            self.target_factor = -2.
            # obstacle avoidance
            self.obstacle_strength = 1.0
            self.obstacle_strength = 1.0
            self.collision_penalty_coef = 0.0
            self.collision_penalty = self.collision_penalty_coef*self.survive_reward 

        
        if self.map_name == 'gate_left':
            # ## Original way
            # self.termination_distance = 1000
            # self.action_strength = 1.0 # only act on body rate
            # self.joint_vel_obs_scaling = 0.1
            self.up_strength = 0.1
            self.heading_strength = 0.5 # reward yaw motion align with lin_vel
            self.lin_strength = -0.5
            self.yaw_strength = 0.0
            self.action_penalty = -0.75
            self.action_change_penalty = -0.75
            # self.ang_vel_penalty = -0.4
            # self.ang_acc_penalty = -0.005
            self.survive_reward = 8.0
            self.height_penalty = -2.0
            self.height_change_penalty = -0.1
            self.jitter_penalty = -0.1
            self.out_map_penalty = -1.5
            self.map_center_penalty = -0.05
            # trajectory
            self.waypoint_strength = 2.0
            self.target_factor = -3.
            # obstacle avoidance
            self.obstacle_strength = 0.5
            self.collision_penalty_coef = 0.0
            self.collision_penalty = self.collision_penalty_coef*self.survive_reward 
        
        if self.map_name == 'gate_right':
            # ## Original way
            # self.termination_distance = 1000
            # self.action_strength = 1.0 # only act on body rate
            # self.joint_vel_obs_scaling = 0.1
            self.up_strength = 0.25
            self.heading_strength = 0.25 # reward yaw motion align with lin_vel
            self.lin_strength = -1.5
            self.lin_vel_rate_penalty = 0.
            self.yaw_strength = 0.0
            self.action_penalty = -1.0
            self.action_change_penalty = -1.0
            self.smooth_penalty = -1.0
            self.smooth_penalty = -1.0
            # self.ang_vel_penalty = -0.4
            # self.ang_acc_penalty = -0.005
            self.survive_reward = 8.0
            self.pose_penalty = -0.5
            self.pose_penalty = -0.5
            self.height_penalty = -2.0
            self.height_change_penalty = -0.5
            self.height_change_penalty = -0.5
            self.jitter_penalty = -0.1
            self.out_map_penalty = -1.5
            self.map_center_penalty = -0.05
            # trajectory
            self.waypoint_strength = 2.0
            self.target_factor = -2.
            self.target_factor = -2.
            # obstacle avoidance
            self.obstacle_strength = 1.0
            self.obstacle_strength = 1.0
            self.collision_penalty_coef = 0.0
            self.collision_penalty = self.collision_penalty_coef*self.survive_reward 

        if self.map_name == 'simple_hover':
            # # Reward based way
            # self.up_strength = 0.5  # Increased reward for staying upright
            # self.heading_strength = 0.0  # Increased reward for good yaw alignment
            # self.lin_strength = 0.5  # Now rewards appropriate linear velocity
            # self.yaw_strength = 0.2  # Reward for yaw stability
            # self.action_reward = 0.3  # Reward for smaller actions
            # self.action_consistency_reward = 0.2  # Reward for consistent actions
            # self.survive_reward = 8.0  # Unchanged base survival reward
            # self.height_reward = 1.5  # Reward for maintaining target height
            # self.height_stability_reward = 0.3  # Reward for stable altitude
            # self.smoothness_reward = 0.2  # Reward for smooth operation
            # self.in_bounds_reward = 1.0  # Reward for staying within map
            # self.map_center_reward = 0.1  # Reward for staying near center
            # # Trajectory
            # self.waypoint_strength = 3.0  # Increased waypoint reward
            # self.target_alignment_factor = 2.0  # Reward for velocity alignment
            # # Obstacle avoidance
            # self.obstacle_clearance_reward = 1.0  # Reward for keeping distance from obstacles
            # self.collision_penalty = 0.0  # Only remaining penalty

            # ## Original way
            # self.termination_distance = 1000
            # self.action_strength = 1.0 # only act on body rate
            # self.joint_vel_obs_scaling = 0.1
            self.up_strength = 0.1
            self.heading_strength = 0.5 # reward yaw motion align with lin_vel
            self.lin_strength = -2.5
            self.yaw_strength = 0.0
            self.action_penalty = -0.75
            self.action_change_penalty = -1.0
            self.smooth_penalty = -1.0
            # self.ang_vel_penalty = -0.4
            # self.ang_acc_penalty = -0.005
            self.survive_reward = 8.0
            self.height_penalty = -2.0
            self.height_change_penalty = -0.5
            self.jitter_penalty = -0.1
            self.out_map_penalty = -1.5
            self.map_center_penalty = -0.05
            # trajectory
            self.waypoint_strength = 2.5
            self.target_factor = -3.
            # obstacle avoidance
            self.obstacle_strength = 0.75
            self.collision_penalty_coef = 0.0
            self.collision_penalty = self.collision_penalty_coef*self.survive_reward 
        
        
        
        if self.map_name == 'clutter':
            # ## Original way
            # self.termination_distance = 1000
            # self.action_strength = 1.0 # only act on body rate
            # self.joint_vel_obs_scaling = 0.1
            self.up_strength = 0.1
            self.heading_strength = 0.3 # reward yaw motion align with lin_vel
            self.lin_strength = -0.4
            self.yaw_strength = 0.0
            self.action_penalty = -0.4
            self.action_change_penalty = -0.4
            # self.ang_vel_penalty = -0.4
            # self.ang_acc_penalty = -0.005
            self.survive_reward = 8.0
            self.height_penalty = -2.0
            self.height_change_penalty = -0.1
            self.jitter_penalty = -0.1
            self.out_map_penalty = -1.5
            self.map_center_penalty = -0.15
            # trajectory
            self.waypoint_strength = 1.5
            self.target_factor = -3.
            # obstacle avoidance
            self.obstacle_strength = 0.3
            self.collision_penalty_coef = 0.0
            self.collision_penalty = self.collision_penalty_coef*self.survive_reward 

        # self.depth_strength = 2.5
        # self.control_base = 0.327 # hover thrust
        self.obst_threshold = 0.25
        self.obst_collision_limit = 0.20
        self.body_rate_threshold = 15
        self.domain_randomization = False

        

        # reward function adjusting
        self.survive_disc_start = 400
        self.lin_rev_start = 1000

        self.dt = 0.05 # outter loop RL policy freaquency
        self.map_limit_fact = 1.25
        self.map_x_min = -1.5; self.map_x_max = 13
        self.map_y_min = -2.5; self.map_y_max = 2.5
        

        self.nerf_count = 0
        self.nerf_freq = 1 # inference nerf every # steps
        self.depth_list = 0
        self.condition_approach = 4
        self.training_stage = 0

        self.traj_planning_count = 0
        

        #-----------------------
        # Set up visual perception net
        self.visual_net = VisualPerceptionNet().to(self.device)
        self.obs_hist_buf = ObsHistBuffer(batch_size=self.num_envs,
                                          vector_dim=self.num_obs,
                                          buffer_size=HISTORY_BUFFER_NUM,
                                          device=self.device,
                                          )


        # Original way pack for record
        self.hyper_parameter = {
            # "termination_distance": self.termination_distance,
            # "action_strength": self.action_strength,
            # "joint_vel_obs_scaling": self.joint_vel_obs_scaling,
            "up_strength": self.up_strength,
            "heading_strength": self.heading_strength,
            "lin_strength": self.lin_strength,
            "obstacle_strength": self.obstacle_strength,
            "obst_collision_limit": self.obst_collision_limit,
            "action_penalty": self.action_penalty,
            "smooth_penalty": self.smooth_penalty,
            # "target_penalty": self.target_penalty,
            "target_factor": self.target_factor,
            "pose_penalty": self.pose_penalty,
            # "ang_vel_penalty": self.ang_vel_penalty,
            # "ang_acc_penalty": self.ang_acc_penalty,
            "obst_threshold": self.obst_threshold,
            "survive_reward": self.survive_reward,
            "height_penalty": self.height_penalty,
            "height_change_penalty": self.height_change_penalty,
            "sim_dt": self.dt,
            "collision_penalty": self.collision_penalty,
            "waypoint_strength": self.waypoint_strength,
            "jitter_penalty":self.jitter_penalty,
            "map_center_penalty": self.map_center_penalty,
            "out_map_penalty": self.out_map_penalty,
            "collision_penalty_coef":self.collision_penalty_coef,
            "condition_approach": self.condition_approach,
            "yaw_strength": self.yaw_strength,
            "action_change_penalty": self.action_change_penalty,
            "min_mass": self.min_mass,
            "mass_range": self.mass_range,
            "min_thrust": self.min_thrust,
            "min_thrust": self.thrust_range,
        }

        # # Reward based way pack for record
        # self.hyper_parameter = {
        #     # Reward strengths
        #     "up_strength": self.up_strength,  # Reward for staying upright
        #     "heading_strength": self.heading_strength,  # Reward for yaw alignment
        #     "lin_strength": self.lin_strength,  # Reward for linear velocity alignment
        #     "obstacle_clearance_reward": self.obstacle_clearance_reward,  # Reward for obstacle distance
        #     "obst_collision_limit": self.obst_collision_limit,  # Distance threshold for collision
        #     "action_reward": self.action_reward,  # Reward for smaller actions
        #     "target_alignment_factor": self.target_alignment_factor,  # Reward for velocity alignment
        #     "obst_threshold": self.obst_threshold,  # Distance threshold for obstacle rewards
        #     "survive_reward": self.survive_reward,  # Base reward for survival
        #     "height_reward": self.height_reward,  # Reward for maintaining target height
        #     "height_stability_reward": self.height_stability_reward,  # Reward for stable altitude
        #     "sim_dt": self.dt,  # Simulation timestep
        #     "collision_penalty": self.collision_penalty,  # Only remaining penalty
        #     "waypoint_strength": self.waypoint_strength,  # Reward for reaching waypoints
        #     "smoothness_reward": self.smoothness_reward,  # Reward for smooth operation
        #     "map_center_reward": self.map_center_reward,  # Reward for staying near center
        #     "in_bounds_reward": self.in_bounds_reward,  # Reward for staying within map
        #     "condition_approach": self.condition_approach,  # Reset condition strategy
        #     "yaw_strength": self.yaw_strength,  # Reward for yaw stability
        #     "action_consistency_reward": self.action_consistency_reward,  # Reward for consistent actions
        # }
        
        #-----------------------
        # set up Visualization
        self.time_stamp = get_time_stamp()
        self.episode_count = 0
        if (self.visualize):
            curr_path = os.getcwd()
            self.save_path = f'{curr_path}/examples/outputs/{self.agent_name}/{self.map_name}/{self.time_stamp}'
            os.makedirs(self.save_path, exist_ok=True)

           
        # recored x, y, depth for test
        self.x_record = []
        self.y_record = []
        self.z_record = []
        self.roll_record = []
        self.pitch_record = []
        self.yaw_record = []
        self.velo_x_record = []
        self.velo_y_record = []
        self.velo_z_record = []
        self.ang_velo_x_record = []
        self.ang_velo_y_record = []
        self.ang_velo_z_record = []
        self.vae_velo_x_record = []
        self.vae_velo_y_record = []
        self.vae_velo_z_record = []
        self.depth_record = []
        self.img_record = []
        self.action_record = np.zeros([self.episode_length,4])
        

    def init_sim(self):
        # self.builder = df.sim.ModelBuilder()
        self.builder = ModelBuilder()
        self.dt = 0.05 # outter loop RL policy freaquency
        self.sim_dt = self.dt
        self.ground = True

        self.num_joint_q = 11 # action(4) + position(3) + pose(4)
        self.num_joint_qd = 10

        self.x_unit_tensor = tu.to_torch([1, 0, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.y_unit_tensor = tu.to_torch([0, 1, 0], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.z_unit_tensor = tu.to_torch([0, 0, 1], dtype=torch.float, device=self.device, requires_grad=False).repeat((self.num_envs, 1))

        if self.map_name == 'gate_mid':
            self.start_rot = np.array([0., 0., 0., 1.]) # (x,y,z,w)
        if self.map_name == 'gate_left':
            self.start_rot = np.array([0., 0., 0., 1.]) # (x,y,z,w)
        if self.map_name == 'gate_right':
            self.start_rot = np.array([0., 0., 0., 1.]) # (x,y,z,w)
        if self.map_name == 'simple_hover':
            self.start_rot = np.array([0., 0., 0., 1.]) # (x,y,z,w)
        if self.map_name == 'clutter':
            # self.start_rot = np.array([0., 0., 0.707, 0.707]) # (x,y,z,w), rpy (0,0,pi/2)
            self.start_rot = np.array([0.0, 0.0, 0.382683, 0.923879]) # (x,y,z,w), rpy (0,0,pi/4)

        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.z_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        # self.mass = torch.ones(self.num_envs, device=self.device)  # All drones have mass = 1 kg
        self.mass = torch.full((self.num_envs,), self.min_mass+0.5*self.mass_range, device=self.device)  # All drones have mass = 1.2 kg
        self.max_thrust = torch.full((self.num_envs,), self.min_thrust+0.5*self.thrust_range, device=self.device)  # All drones have max_thrust = 23 N
        self.hover_thrust = self.mass * 9.81  # Hover thrust for each drone
        inertia = torch.tensor([0.008, 0.01, 0.025], device=self.device)  # Diagonal inertia components
        self.inertia = torch.diag(inertia).unsqueeze(0).repeat(self.num_envs, 1, 1)  # Repeat for all drones
        link_length = 0.15  # meters
        Kp = torch.tensor([0.8, 1., 2.5], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # Repeat for all drones
        Kd = torch.tensor([0.001, 0.001, 0.002], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # Repeat for all drones

        # Initialize the QuadrotorSimulator
        self.quad_dynamics = QuadrotorSimulator(
            mass=self.mass,  # Shape: (batch_size,)
            inertia=self.inertia,  # Shape: (batch_size, 3, 3)
            link_length=link_length,  # Scalar
            Kp=Kp,  # Shape: (batch_size, 3)
            Kd=Kd,  # Shape: (batch_size, 3)
            freq=200.0,  # Scalar
            max_thrust=self.max_thrust,  # Shape: (batch_size,)
            total_time=self.sim_dt,  # Scalar
            rotor_noise_std=0.01,  # Scalar
            br_noise_std=0.01  # Scalar
        )
        
        if self.map_name == 'gate_mid':
            # start_pos_x = -0.5; start_pos_y = 0.5
            # start_pos_x = -1.0; start_pos_y = -1.0
            start_pos_x = 0.0; start_pos_y = 0.0
            self.start_body_rate = [0., 0., 0.]
        if self.map_name == 'gate_left':
            start_pos_x = 0.0; start_pos_y = 0.0
            self.start_body_rate = [0., 0., 0.]
        if self.map_name == 'gate_right':
            start_pos_x = 0.0; start_pos_y = 0.0
            self.start_body_rate = [0., 0., 0.]
        if self.map_name == 'simple_hover':
            start_pos_x = -0.0; start_pos_y = -0.0
            self.start_body_rate = [0., 0., 0.]
        if self.map_name == 'clutter':
            start_pos_x = -0.5; start_pos_y = 1.0
            self.start_body_rate = [0., 0., 0.]

        self.start_height = 1.4
        self.start_pos = []
        # self.start_body_rate = [0., 0., 0.] # r,p,y
        self.start_norm_thrust = [(self.hover_thrust / self.max_thrust).clone().detach().cpu().numpy()[0]]
        self.control_base = self.start_norm_thrust[0]
        self.start_action = self.start_body_rate + self.start_norm_thrust
        
        
        # if self.visualize:
        #     self.env_dist = 2.5
        # else:
        #     self.env_dist = 0. # set to zero for training for numerical consistency
        

        for i in range(self.num_environments):
            # start_pos_y = i*self.env_dist
            start_pos = [start_pos_x, start_pos_y, self.start_height, ]
            self.start_pos.append(start_pos)

        # start_x variables for reseting drones       
        self.start_pos = tu.to_torch(self.start_pos, device=self.device)
        self.start_pos_backup = self.start_pos.clone()
        self.start_joint_q = tu.to_torch(self.start_action, device=self.device)
        self.start_joint_target = tu.to_torch(self.start_action, device=self.device)
        self.prev_thrust = torch.ones([self.num_envs, 1], device=self.device) * (self.hover_thrust / self.max_thrust)
        
        # initialize action with hover thrust
        self.actions = self.start_joint_q.repeat(self.num_envs,1).clone()
        self.prev_actions = self.actions.clone()
        self.prev_prev_actions = self.actions.clone()
        # initialize state variables
        self.state_joint_q = torch.zeros([self.num_envs, 7], device=self.device) # pos + quat
        self.state_joint_qd = torch.zeros([self.num_envs, 6], device=self.device) # lin_velo + ang_velo
        self.state_joint_qdd = torch.zeros([self.num_envs, 6], device=self.device) # lin_acc + ang_acc
        
        self.latent_vect = torch.zeros([self.num_envs, LATENT_VECT_NUM], device=self.device, dtype = torch.float)
        self.lin_vel_vae = torch.zeros([self.num_envs, 3], device=self.device, dtype = torch.float)
        self.prev_lin_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype = torch.float)

    
    def step(self, actions, vae_info, vel_net_info):
        self.latent_vect, _ = vae_info[0].clone().detach(), vae_info[1].clone().detach()
        self.lin_vel_vae = vel_net_info[0].clone().detach()

        # prepare vae data
        # original
        # self.obs_hist_buf.update(self.obs_buf)
        # ablate latent vec for vae
        self.obs_hist_buf.update(self.vae_obs_buf)
        obs_hist = self.obs_hist_buf.get_concatenated().clone().detach()
        obs_vel = self.privilege_obs_buf[:, 3:6].clone().detach()

        actions = actions.view((self.num_envs, self.num_actions))
        body_rate_cols = torch.clip(actions[:, 0:3], -1., 1.) * 0.5

        # # adding delay for br
        # prev_body_rate = self.prev_actions[:, 0:3].clone().detach()
        # body_rate_cols = 0.7*(torch.clip(actions[:, 0:3], -1., 1.) * 0.5) + 0.3*prev_body_rate
        # body_rate_cols = torch.clip(body_rate_cols, -0.5, 0.5)

        # thrust_col = torch.clip(actions[:, 3:],-1., 1.) * 0.2 + self.control_base
        # thrust_col = torch.clip(actions[:, 3:],-1., 1.) * 0.2 + self.control_base
        prev_thrust = self.prev_actions[:, -1].unsqueeze(-1).clone().detach()
        thrust_col = 0.7*((torch.clip(actions[:, 3:],-1., 1.) + 1) * 0.275) + 0.3*prev_thrust # rough simulation of motor delay
        thrust_col = torch.clip(thrust_col, 0.30, 0.55) # 0 to 0.55

        actions = torch.cat([body_rate_cols, thrust_col], dim=1)
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()
        
        # # test direction of rpy
        # actions = torch.zeros_like(actions)
        # actions[:, 3] = 0.4
        # actions[:, 0] = 0.25 # positive roll
        # actions[:, 1] = -0.25 # positive pitch
        # actions[:, 2] = 0.25 # positive yaw

        self.actions = actions # normalized
        control_input = (actions[:, 0:3], actions[:, 3])
        
        torso_pos = self.state_joint_q.view(self.num_envs, -1)[:, 0:3] # (x, y, z)
        torso_quat = self.state_joint_q.view(self.num_envs, -1)[:, 3:7] # rotated 90 deg
        lin_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 3:6] # joint_qd rot has 3 entries
        ang_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 0:3]
        lin_acc = self.state_joint_qdd.view(self.num_envs, -1)[:, 3:6] # joint_qd rot has 3 entries
        ang_acc = self.state_joint_qdd.view(self.num_envs, -1)[:, 0:3]
        self.prev_lin_vel = self.obs_buf[:, 3:6].clone()
        # torso_quat = tu.quat_mul(torso_rot, self.inv_start_rot)

        # if self.episode_count > 70:
        #     print(self.episode_count)

        self.time_report.start_timer("dynamic simulation")
        new_position, new_linear_velocity, new_angular_velocity, new_quaternion, new_linear_acceleration, new_angular_acceleration = self.quad_dynamics.run_simulation(
        position=torso_pos,
        velocity=lin_vel,
        orientation=torso_quat[:, [3,0,1,2]],
        angular_velocity=ang_vel,
        control_input=control_input
    )
        self.time_report.end_timer("dynamic simulation")
        # set new state back
        if self.no_grad:
            new_position = new_position.detach()
            new_quaternion = new_quaternion.detach()
            new_linear_velocity = new_linear_velocity.detach()
            new_angular_velocity = new_angular_velocity.detach()


        self.state_joint_q.view(self.num_envs, -1)[:, 0:3] = new_position
        self.state_joint_q.view(self.num_envs, -1)[:, 3:7] = new_quaternion[:, [1,2,3,0]].clone()
        self.state_joint_qd.view(self.num_envs, -1)[:, 3:6] = new_linear_velocity
        self.state_joint_qd.view(self.num_envs, -1)[:, 0:3] = new_angular_velocity
        self.state_joint_qdd.view(self.num_envs, -1)[:, 3:6] = new_linear_acceleration.clone().detach()
        self.state_joint_qdd.view(self.num_envs, -1)[:, 0:3] = new_angular_acceleration.clone().detach()
        # joint_qdd = self.state_joint_qdd.view(self.num_envs, -1).clone()  # Clone the tensor to avoid in-place operation
        # joint_qdd[:, 3:6] = new_linear_acceleration
        # joint_qdd[:, 0:3] = new_angular_acceleration
        # self.state_joint_qdd = joint_qdd.view_as(self.state_joint_qdd)  # Assign it back

        self.sim_time += self.sim_dt
        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations()
        self.calculateReward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.no_grad == False:
            self.obs_buf_before_reset = self.obs_buf.clone()
            self.privilege_obs_buf_before_reset = self.privilege_obs_buf.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'privilege_obs_before_reset': self.privilege_obs_buf_before_reset,
                'episode_end': self.termination_buf,
                }

        if len(env_ids) > 0:
           self.reset(env_ids)

        # self.render()
        if (torch.isnan(obs_hist).any() | torch.isinf(obs_hist).any()):
            print('obs hist nan')
            obs_hist = torch.nan_to_num(obs_hist, nan=0.0, posinf=1e3, neginf=-1e3)
        if (torch.isnan(obs_vel).any() | torch.isinf(obs_vel).any()):
            print('obs vel nan')
            obs_vel = torch.nan_to_num(obs_vel, nan=0.0, posinf=1e3, neginf=-1e3)
        

        return self.obs_buf, self.privilege_obs_buf, obs_hist, obs_vel, self.rew_buf, self.reset_buf, self.extras
    
    def change_curriculum(self, type):
        if type == 0:
            self.start_pos = self.start_pos_backup.clone()
        elif type == 1:
            self.start_pos = self.reward_wp[0].repeat(self.num_envs,1)
            return self.start_pos[0]
        elif type == 2:
            self.start_pos = self.reward_wp[1].repeat(self.num_envs,1)
            return self.start_pos[0]
        elif type == 3:
            self.start_rot = quat_from_axis_angle((0.707, -0.707, 0.0), -math.pi*0.5)
            self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)
            return self.start_rotation

    def reset(self, env_ids = None, force_reset = True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            if not self.visualize and self.domain_randomization:

                # Randomize mass, max_thrust, and inertia for the specified environments
                num_reset_envs = len(env_ids)
                mass_with_noise = torch.rand(num_reset_envs, device=self.device) * self.mass_range + self.min_mass  # Uniform noise between 1.0 to 1.2
                self.mass[env_ids] = mass_with_noise
                max_thrust_with_noise = torch.rand(num_reset_envs, device=self.device) * self.thrust_range + self.min_thrust  # Uniform noise between 24 and 26
                self.max_thrust[env_ids] = max_thrust_with_noise
                inertia = torch.tensor([0.008, 0.01, 0.025], device=self.device)  # Base inertia
                inertial_noise = (torch.rand(num_reset_envs, 3, device=self.device) - 0.5) * 2 * 0.2 * inertia  # ±20% noise
                randomized_inertia = inertia + inertial_noise
                self.inertia[env_ids] = torch.diag_embed(randomized_inertia)  # Shape: (num_reset_envs, 3, 3)
                self.hover_thrust[env_ids] = self.mass[env_ids] * 9.81

                # Reinitialize QuadrotorSimulator with updated parameters
                self.quad_dynamics = QuadrotorSimulator(
                    mass=self.mass,  # Shape: (batch_size,)
                    inertia=self.inertia,  # Shape: (batch_size, 3, 3)
                    link_length=0.15,  # Scalar
                    Kp=torch.tensor([0.8, 1., 2.5], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),  # Shape: (batch_size, 3)
                    Kd=torch.tensor([0.001, 0.001, 0.002], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),  # Shape: (batch_size, 3)
                    freq=200.0,  # Scalar
                    max_thrust=self.max_thrust,  # Shape: (batch_size,)
                    total_time=self.sim_dt,  # Scalar
                    rotor_noise_std=0.01,  # Scalar
                    br_noise_std=0.01  # Scalar
                )

                


                # self.mass = random.uniform(0.8,1.2)
                # self.max_thrust = random.uniform(17,28) # randomized max_thrust
                # self.hover_thrust = self.mass * 9.81 
                # inertia = torch.tensor([0.002, 0.002, 0.004])  # Diagonal inertia tensor components
                # # inertial_noise = (torch.rand_like(inertia) - 0.5) * 2 * 0.2 * inertia
                # inertial_noise = torch.zeros_like(inertia)
                # randomized_inertia = inertia + inertial_noise
                # link_length = 0.15  # meters
                # self.quad_dynamics = QuadrotorSimulator(
                #                         mass=self.mass,
                #                         inertia=torch.diag(randomized_inertia),
                #                         link_length=link_length,
                #                         Kp=torch.tensor([0.5, 0.5, 1.0]),
                #                         Kd=torch.tensor([0.03, 0.03, 0.03]),
                #                         freq=200.0,
                #                         max_thrust=self.max_thrust,  # Maximum thrust in Newtons
                #                         total_time=self.sim_dt,  # Total simulation time in seconds
                #                         rotor_noise_std=0.025,  # Standard deviation for rotor control noise
                #                         br_noise_std=0.025
                #                     )
        
                

                self.start_norm_thrust = [(self.hover_thrust / self.max_thrust).clone().detach().cpu().numpy()[0]]
                self.control_base = self.start_norm_thrust[0]
                self.start_action = self.start_body_rate + self.start_norm_thrust
                self.start_joint_q = tu.to_torch(self.start_action, device=self.device)
            
            # clone the state to avoid gradient error
            self.state_joint_q = self.state_joint_q.clone()
            self.state_joint_qd = self.state_joint_qd.clone()
            self.state_joint_qdd = self.state_joint_qdd.clone()

            # fixed start state
            self.state_joint_q[env_ids, 0:3] = self.start_pos[env_ids, :].clone()
            self.state_joint_q.view(self.num_envs, -1)[env_ids, 3:7] = self.start_rotation.clone()
            # self.state_joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.start_joint_q.clone()
            self.state_joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.
            self.state_joint_qdd.view(self.num_envs, -1)[env_ids, :] = 0.
            self.actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()
            self.prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()
            self.prev_prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()

            # randomization
            if not self.visualize and self.stochastic_init:
                self.state_joint_q.view(self.num_envs, -1)[env_ids, 0:3] = self.state_joint_q.view(self.num_envs, -1)[env_ids, 0:3] + 0.5 * (torch.rand(size=(len(env_ids), 3), device=self.device) - 0.5) * 2.
                angle = (torch.rand(len(env_ids), device = self.device) - 0.5) * np.pi / 12.
                axis = torch.nn.functional.normalize(torch.rand((len(env_ids), 3), device = self.device) - 0.5)
                self.state_joint_q.view(self.num_envs, -1)[env_ids, 3:7] = tu.quat_mul(self.state_joint_q.view(self.num_envs, -1)[env_ids, 3:7], tu.quat_from_angle_axis(angle, axis))
                # self.state_joint_q.view(self.num_envs, -1)[env_ids, 7:] = self.state_joint_q.view(self.num_envs, -1)[env_ids, 7:] + 0.2 * (torch.rand(size=(len(env_ids), self.num_joint_q - 7), device = self.device) - 0.5) * 2.
                self.state_joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), 6), device=self.device) - 0.5)
                self.state_joint_qdd.view(self.num_envs, -1)[env_ids, :] = 0.05 * (torch.rand(size=(len(env_ids), 6), device=self.device) - 0.5)
                self.actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone() + 0.05 * (torch.rand(size=(len(env_ids), 4), device=self.device))
                self.prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()
                self.prev_prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()
            
            

            # clear action
            # self.actions = self.actions.clone()
            # self.actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device = self.device, dtype = torch.float)
            # self.prev_actions = self.prev_actions.clone()
            # self.prev_actions[env_ids, :] = torch.zeros((len(env_ids), self.num_actions), device = self.device, dtype = torch.float)

            # clear VAE variables
            self.latent_vect = torch.zeros([self.num_envs, LATENT_VECT_NUM], device=self.device, dtype = torch.float)
            self.lin_vel_vae = torch.zeros([self.num_envs, 3], device=self.device, dtype = torch.float)

            self.progress_buf[env_ids] = 0
            self.calculateObservations()

        return self.obs_buf
    
    '''
    cut off the gradient from the current state to previous states
    '''
    def clear_grad(self, checkpoint = None):
        with torch.no_grad():
            if checkpoint is None:
                checkpoint = {}
                checkpoint['joint_q'] = self.state_joint_q.clone()
                checkpoint['joint_qd'] = self.state_joint_qd.clone()
                checkpoint['actions'] = self.actions.clone()
                checkpoint['prev_actions'] = self.prev_actions.clone()
                checkpoint['prev_prev_actions'] = self.prev_prev_actions.clone()
                checkpoint['progress_buf'] = self.progress_buf.clone()
                checkpoint['latent_vect'] = self.latent_vect.clone()
                checkpoint['lin_vel_vae'] = self.lin_vel_vae.clone()
                checkpoint['prev_lin_vel'] = self.prev_lin_vel

            current_joint_q = checkpoint['joint_q'].clone()
            current_joint_qd = checkpoint['joint_qd'].clone()
            # self.state = self.model.state()
            self.state_joint_q = current_joint_q
            self.state_joint_qd = current_joint_qd
            self.actions = checkpoint['actions'].clone()
            self.prev_actions = checkpoint['prev_actions'].clone()
            self.prev_prev_actions = checkpoint['prev_prev_actions'].clone()
            self.progress_buf = checkpoint['progress_buf'].clone()
            self.latent_vect = checkpoint['latent_vect'].clone()
            self.lin_vel_vae = checkpoint['lin_vel_vae'].clone()
            self.prev_lin_vel = checkpoint['prev_lin_vel'].clone()

    '''
    This function starts collecting a new trajectory from the current states but cuts off the computation graph to the previous states.
    It has to be called every time the algorithm starts an episode and it returns the observation vectors
    '''
    def initialize_trajectory(self):
        self.clear_grad()
        self.calculateObservations()

        return self.obs_buf, self.privilege_obs_buf

    def get_checkpoint(self):
        checkpoint = {}
        checkpoint['joint_q'] = self.state_joint_q.clone()
        checkpoint['joint_qd'] = self.state_joint_qd.clone()
        checkpoint['actions'] = self.actions.clone()
        checkpoint['prev_actions'] = self.prev_actions.clone()
        checkpoint['prev_prev_actions'] = self.prev_prev_actions.clone()
        checkpoint['progress_buf'] = self.progress_buf.clone()
        checkpoint['latent_vect'] = self.latent_vect.clone()
        checkpoint['lin_vel_vae'] = self.lin_vel_vae.clone()
        checkpoint['prev_lin_vel'] = self.prev_lin_vel.clone()

        return checkpoint
    
    def process_GS_data(self, depth_list, nerf_img):
        # RGB ablation
        nerf_img = torch.zeros_like(nerf_img)

        # min depth from nerf
        batch,H,W,ch = depth_list.shape
        depth_list_up = depth_list[:,0:int(H/2),:,:]
        self.depth_list = torch.abs(torch.amin(depth_list_up,dim=(1,2,3))) 
        self.depth_list = self.depth_list.unsqueeze(1)
        self.depth_list = self.depth_list.to(device=self.device)
        # rgb from nerf
        input_tensor = nerf_img.permute(0, 3, 1, 2)
        depth_tensor = depth_list.permute(0, 3, 1, 2)  # Make sure depth_list is [batch_size, channels, height, width]
        depth_tensor = torch.clip(depth_tensor, 0.0, 2.5) # match the realsense hardware limit
        
        # Concatenate RGB image and depth data along the channel dimension
        combined_tensor = torch.cat([input_tensor, depth_tensor], dim=1)
        # Resize to match SqueezeNet input size
        resize = nn.AdaptiveAvgPool2d((224, 224))
        combined_tensor = resize(combined_tensor)
        combined_tensor = combined_tensor.to(self.device)
        self.visual_info = self.visual_net(combined_tensor)
        self.visual_info = self.visual_info.detach()

        # # Only use RGB
        # resize = nn.AdaptiveAvgPool2d((224, 224))
        # input_tensor = resize(input_tensor)
        # input_tensor = input_tensor.to(self.device)
        # self.visual_info = self.visual_net(input_tensor)
        # self.visual_info = self.visual_info.detach()
        


    def calculateObservations(self):
        torso_pos = self.state_joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_quat = self.state_joint_q.view(self.num_envs, -1)[:, 3:7] # (x,y,z,w)
        lin_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 3:6] # joint_qd rot has 3 entries
        ang_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
        lin_vel_new = lin_vel - torch.cross(torso_pos, ang_vel, dim=-1).detach().clone()
        # lin_vel = torch.clamp(lin_vel.clone(), min=-1e6, max=1e6)
        # self.lin_vel_vae = torch.clamp(self.lin_vel_vae.clone(), min=-1e6, max=1e6)

        to_target =  + self.start_pos - torso_pos
        to_target[:, 2] = 0.0
        
        # target_dirs = tu.normalize(to_target)
        target_dirs = tu.normalize(lin_vel[:, 0:2].clone())
        up_vec = tu.quat_rotate(torso_quat.clone(), self.up_vec)
        up_vec_ablation = torch.zeros_like(up_vec)
        # heading_vec = tu.quat_rotate(torso_quat, self.heading_vec)
        rpy = quaternion_to_euler(torso_quat[:, [3,0,2,1]]) # input is (w,x,z,y)
        heading_vec = torch.cat([torch.cos(rpy[:,2].unsqueeze(-1)), torch.sin(rpy[:,2].unsqueeze(-1))], dim=1) 

        # Parallized implementation
        # gs_quat = torso_quat # (x, y, z, w)
        gs_pos = torso_pos + self.gs_origin_offset # (x, y, z)
        gs_pos[:, 1] = -gs_pos[:, 1]
        gs_pos[:, 2] = -gs_pos[:, 2]
        gs_pose = torch.cat([gs_pos, torch.zeros([self.num_envs, 3], device = self.device), torso_quat], dim=-1)
        rpy_data = rpy.clone().detach().cpu().numpy()


        if not self.visualize:
            # Nerf info processing
            if self.nerf_count % self.nerf_freq == 0: # infer nerf every nerf_freq steps
                self.time_report.start_timer("3D GS inference")
                depth_list, nerf_img = self.gs.render(gs_pose) # (batch_size,H,W,1/3)
                self.process_GS_data(depth_list, nerf_img)
                self.time_report.end_timer("3D GS inference")
            self.nerf_count += 1            
            
        # Draw record plot for data visualization
        if self.visualize:
            img_transform = Resize((360, 640), antialias=True)

            # ---------------------------------------
            # Normal test mode
            # NeRF data
            depth_list, nerf_img = self.gs.render(gs_pose) # (batch_size,H,W,1/3)
            self.process_GS_data(depth_list, nerf_img)

            # Visualization
            # rgb from nerf
            nerf_img = torch.permute(nerf_img[0], (2,0,1))
            img = img_transform(nerf_img)

            pos_data = torso_pos.clone()
            pos_data = pos_data.detach().cpu().numpy()
            depth_data = self.depth_list.clone()
            depth_data = depth_data.detach().cpu().numpy()
            action_data = self.actions.clone().detach().cpu().numpy()
            ang_vel_data = ang_vel.clone().detach().cpu().numpy()
            # velo_data = lin_vel_new.clone().detach().cpu().numpy()
            vae_velo_data = self.lin_vel_vae.clone().detach().cpu().numpy()
            velo_data = lin_vel.clone().detach().cpu().numpy()

            self.x_record.append(pos_data[0, 0])
            self.y_record.append(pos_data[0, 1])
            self.z_record.append(pos_data[0, 2])
            self.roll_record.append(rpy_data[0, 0])
            self.pitch_record.append(rpy_data[0, 1])
            self.yaw_record.append(rpy_data[0, 2])
            self.velo_x_record.append(velo_data[0, 0])
            self.velo_y_record.append(velo_data[0, 1])
            self.velo_z_record.append(velo_data[0, 2])
            self.ang_velo_x_record.append(ang_vel_data[0, 0])
            self.ang_velo_y_record.append(ang_vel_data[0, 1])
            self.ang_velo_z_record.append(ang_vel_data[0, 2])
            self.vae_velo_x_record.append(vae_velo_data[0, 0])
            self.vae_velo_y_record.append(vae_velo_data[0, 1])
            self.vae_velo_z_record.append(vae_velo_data[0, 2])
            self.depth_record.append(depth_data[0]/2)
            self.img_record.append(img)
            if self.episode_count < self.episode_length:
                self.action_record[self.episode_count,:] = action_data
            self.episode_count += 1

        
        latent_abalation = torch.zeros_like(self.latent_vect)
        visual_ablation = torch.zeros_like(self.visual_info)
        torso_rot_ablation = torch.zeros_like(torso_quat)
        ang_vel_ablation = torch.zeros_like(ang_vel)
        lin_vel_ablation = torch.zeros_like(lin_vel)
        lin_acceleration = self.state_joint_qdd.view(self.num_envs, -1)[:, 3:6]
        ang_acceleration = self.state_joint_qdd.view(self.num_envs, -1)[:, 0:3]
        torso_pos_ablation = torch.zeros_like(torso_pos)
        self.privilege_obs_buf = torch.cat([
                                torso_pos[:, :], # 0:3 
                                # lin_vel_new, # 3:6
                                lin_vel,
                                lin_acceleration, # acc 6:9 
                                torso_quat, # 9:13
                                ang_vel, # 13:16
                                # up_vec[:, 1:2], # 16
                                up_vec_ablation[:, 1:2],
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 17
                                self.actions, # 18:22
                                self.prev_actions, # 22:26
                                self.depth_list, # 26
                                self.visual_info, # 27:51
                                # self.latent_vect, # 51:67
                                latent_abalation,
                                # ang_acceleration, # 59:62
                                self.lin_vel_vae,
                                ], 
                                dim = -1)
        
        # adding noise
        torso_pos_noise = self.obs_noise_level * (torch.rand_like(torso_pos)-0.5)
        lin_vel_noise = self.obs_noise_level * (torch.rand_like(lin_vel) - 0.5)
        visual_noise = self.obs_noise_level * (torch.rand_like(self.visual_info) - 0.5)
        torso_quat_noise = self.obs_noise_level * (torch.rand_like(torso_quat) - 0.5)
        if self.training_stage == 0:
            # # add noise
            self.obs_buf = torch.cat([
                                    (torso_pos[:, 2]+torso_pos_noise[:, 2]).unsqueeze(-1), # z-pos 0
                                    # torso_pos_ablation[:, 2].unsqueeze(-1), # z-pos 0

                                    # lin_vel_new[:, 2].unsqueeze(-1), # z-vel 1
                                    (lin_vel[:, 2]+lin_vel_noise[:, 2]).unsqueeze(-1),

                                    # lin_acceleration[:, 2].unsqueeze(-1), # z-acc 2
                                    torso_pos_ablation[:, 2].unsqueeze(-1),

                                    torso_quat+torso_quat_noise, # 3:7
                                    # torso_rot_ablation,

                                    # ang_vel, # 7:10
                                    ang_vel_ablation,

                                    # up_vec[:, 1:2], # 10
                                    up_vec_ablation[:, 1:2],

                                    self.actions, # 11:15
                                    self.prev_actions, # 15:19

                                    # self.lin_vel_vae, # 19:22
                                    lin_vel,
                                    # lin_vel_ablation,

                                    self.visual_info+visual_noise, # 22:46
                                    # visual_ablation,
                                    
                                    latent_abalation,
                                    # self.latent_vect, # 46:62
                                    ], 
                                    dim = -1)
            
            # # noise-free version
            # self.obs_buf = torch.cat([
            #                         (torso_pos[:, 2]).unsqueeze(-1), # z-pos 0
            #                         # torso_pos_ablation[:, 2].unsqueeze(-1), # z-pos 0

            #                         # lin_vel_new[:, 2].unsqueeze(-1), # z-vel 1
            #                         (lin_vel[:, 2]).unsqueeze(-1),

            #                         # lin_acceleration[:, 2].unsqueeze(-1), # z-acc 2
            #                         torso_pos_ablation[:, 2].unsqueeze(-1),

            #                         torso_quat+torso_quat_noise, # 3:7
            #                         # torso_rot_ablation,

            #                         # ang_vel, # 7:10
            #                         ang_vel_ablation,

            #                         # up_vec[:, 1:2], # 10
            #                         up_vec_ablation[:, 1:2],

            #                         self.actions, # 11:15
            #                         self.prev_actions, # 15:19
            #                         # self.lin_vel_vae, # 19:22
            #                         lin_vel,
            #                         self.visual_info, # 22:46
            #                         # latent_abalation
            #                         self.latent_vect, # 46:62
            #                         ], 
            #                         dim = -1)
        else:
            self.obs_buf = torch.cat([
                                    torso_pos[:, 2].unsqueeze(-1), # z-pos 0
                                    # lin_vel_new[:, 2].unsqueeze(-1), # z-vel 1
                                    lin_vel[:, 2].unsqueeze(-1),
                                    lin_acceleration[:, 2].unsqueeze(-1), # z-acc 2
                                    torso_quat, # 3:7
                                    # torso_rot_ablation,
                                    ang_vel, # 7:10
                                    # up_vec[:, 1:2], # 10
                                    up_vec_ablation[:, 1:2],
                                    self.actions, # 11:15
                                    self.prev_actions, # 15:19

                                    self.lin_vel_vae, # 19:22
                                    # lin_vel,

                                    self.visual_info, # 22:46
                                    # latent_abalation
                                    self.latent_vect, # 46:62
                                    ], 
                                    dim = -1)
        
        self.vae_obs_buf = torch.cat([
                                    (torso_pos[:, 2]+torso_pos_noise[:, 2]).unsqueeze(-1), # z-pos 0
                                    # torso_pos_ablation[:, 2].unsqueeze(-1), # z-pos 0

                                    # lin_vel_new[:, 2].unsqueeze(-1), # z-vel 1
                                    (lin_vel[:, 2]+lin_vel_noise[:, 2]).unsqueeze(-1),

                                    # lin_acceleration[:, 2].unsqueeze(-1), # z-acc 2
                                    torso_pos_ablation[:, 2].unsqueeze(-1),

                                    torso_quat+torso_quat_noise, # 3:7
                                    # torso_rot_ablation,

                                    # ang_vel, # 7:10
                                    ang_vel_ablation,

                                    # up_vec[:, 1:2], # 10
                                    up_vec_ablation[:, 1:2],

                                    self.actions, # 11:15
                                    self.prev_actions, # 15:19

                                    # self.lin_vel_vae, # 19:22
                                    lin_vel,
                                    # lin_vel_ablation,

                                    self.visual_info+visual_noise, # 22:46
                                    # visual_ablation,

                                    latent_abalation
                                    # self.latent_vect, # 46:62
                                    ], 
                                    dim = -1)
        
        # ugly fix of simulation (usually z_acc inf) 
        if (torch.isnan(self.obs_buf).any() | torch.isinf(self.obs_buf).any()):
            print('nan')
            self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=1e3, neginf=-1e3)

        if (torch.isnan(self.privilege_obs_buf).any() | torch.isinf(self.privilege_obs_buf).any()):
            print('nan')
            self.privilege_obs_buf = torch.nan_to_num(self.privilege_obs_buf, nan=0.0, posinf=1e3, neginf=-1e3)

        if (torch.isnan(self.vae_obs_buf).any() | torch.isinf(self.vae_obs_buf).any()):
            print('nan')
            self.vae_obs_buf = torch.nan_to_num(self.vae_obs_buf, nan=0.0, posinf=1e3, neginf=-1e3)

        # # ## used for adding breakpoint
        # print(self.episode_count)
        # if self.episode_count % 40 == 0:
        #     print(self.episode_count)

    def calculateReward(self):
        # torso_pos = self.state_joint_q.view(self.num_envs, -1)[:, 0:3]
        # torso_rot = self.state_joint_q.view(self.num_envs, -1)[:, 3:7]
        torso_pos = self.privilege_obs_buf[:, 0:3]
        # torso_rot = self.obs_buf[:, 3:7]
        torso_quat = self.obs_buf[:, 3:7] # (x,y,z,w)
        drone_rot = torso_quat # (x, y, z, w)
        drone_pos = torso_pos + self.point_could_origin_offset # in point cloud dimension
        drone_target = self.target + self.point_could_origin_offset

        # make yaw angle align with velocity
        # target_dirs = tu.normalize(self.privilege_obs_buf[:, [3,5]])
        # rpy = quaternion_to_euler(torso_quat[:, [3,0,2,1]]).detach() # input is (w,x,z,y)
        # heading_vec = torch.cat([torch.cos(rpy[:,2].unsqueeze(-1)), torch.sin(rpy[:,2].unsqueeze(-1))], dim=1) 
        # yaw_alignment = (heading_vec * target_dirs).sum(dim = -1)

        # use quat directly
        target_dirs = tu.normalize(self.privilege_obs_buf[:, 3:5])
        heading_vec = quaternion_yaw_forward(torso_quat)  # Extract (x, y) forward direction, input is (x, y, z, w)
        yaw_alignment = (heading_vec * target_dirs).sum(dim=-1) 
        
        # # Reward tracking reference trajectory
        # # Compute ref_x for all environments
        # ref_x = self.privilege_obs_buf[:, 0] + self.point_could_origin_offset[0, 0]  # Shape: [num_envs]
        # ref_traj_x = self.ref_traj[:, 0]  # Shape: [ref_traj_length]
        # diff = ref_traj_x.unsqueeze(0) - ref_x.unsqueeze(1)  # Shape: [num_envs, ref_traj_length]
        # mask = diff > 0.5  # Shape: [num_envs, ref_traj_length]
        # diff_masked = torch.where(mask, diff, float('inf'))  # Shape: [num_envs, ref_traj_length]
        # min_diffs, min_indices = torch.min(diff_masked, dim=1)  # Shapes: [num_envs], [num_envs]
        # no_valid_indices = torch.isinf(min_diffs)  # Shape: [num_envs]
        # target_list = torch.empty((self.num_envs, 3), device=self.device)
        # default_target = self.target[0] + self.point_could_origin_offset[0]  # Shape: [3]
        # target_list[:] = default_target
        # valid_indices = ~no_valid_indices
        # selected_indices = min_indices[valid_indices]
        # targets = self.ref_traj[selected_indices]  # Shape: [num_valid_envs, 3]
        # target_list[valid_indices] = targets
        # target_list = target_list - self.point_could_origin_offset

        # desire_velo_norm = (target_list - self.privilege_obs_buf[:, 0:3]) / torch.clamp(torch.norm(target_list - self.privilege_obs_buf[:, 0:3], dim=1, keepdim=True), min=1e-6)
        # curr_velo_norm = self.obs_buf[:, 19:22] / torch.clamp(torch.norm(self.obs_buf[:, 19:22], dim=1, keepdim=True), min=1e-2)
        # # curr_velo_norm = self.privilege_obs_buf[:, 3:6] / torch.clamp(torch.norm(self.privilege_obs_buf[:, 3:6], dim=1, keepdim=True), min=1e-2)
        # # velo_dist = torch.linalg.norm(curr_velo_norm[:, [0,2]] - desire_velo_norm[:, [0,2]], dim=1)
        # velo_dist = torch.linalg.norm(curr_velo_norm - desire_velo_norm, dim=1)
        # target_reward = velo_dist * self.target_factor



        # ## Original way
        survive_reward = self.survive_reward
        up_reward = self.up_strength * self.obs_buf[:, 10]
        heading_reward = self.heading_strength * yaw_alignment
        
        # ang_vel_penalty = torch.sum(torch.square(self.obs_buf[:, 7:10]), dim = -1) * self.ang_vel_penalty
        # ang_vel_penalty = torch.linalg.norm(self.obs_buf[:, 7:10], dim=1) * self.ang_vel_penalty
        # ang_vel_penalty = torch.linalg.norm(self.obs_buf[:, 7:9], dim=1) * self.ang_vel_penalty # not penalty yaw
        # ang_acc_penalty = torch.linalg.norm(self.obs_buf[:, 51:54], dim=1) * self.ang_acc_penalty

        # action_penalty = torch.sum(torch.square(self.obs_buf[:, 11:15] - torch.tensor([0,0,0,0.38], device=self.device).repeat((self.num_envs,1))),
        #                             dim = -1) * self.action_penalty # penalty on force
        # action_penalty += torch.square(self.obs_buf[:, 14]) * self.yaw_strength
        # action_change_penalty = torch.sum(torch.square((self.obs_buf[:, 11:15] - self.obs_buf[:, 15:19])), dim = -1) * self.action_change_penalty # penalty on force
        # smooth_penalty = torch.sum(torch.square((self.obs_buf[:, 11:15] - 2*self.obs_buf[:, 15:19] + self.prev_prev_actions)), dim = -1) * self.smooth_penalty
        
        action_penalty = torch.sum(torch.square(self.obs_buf[:, 11:14]),
                                    dim = -1) * self.action_penalty # penalty on force
        action_penalty += torch.square(self.obs_buf[:, 13]) * self.yaw_strength
        action_penalty += torch.sum(torch.square(self.obs_buf[:, 14]-0.42)) * (2*self.action_penalty) 
        action_change_penalty = torch.sum(torch.square((self.obs_buf[:, 11:14] - self.obs_buf[:, 15:18])), dim = -1) * self.action_change_penalty # penalty on force
        action_change_penalty += torch.sum(torch.square(self.obs_buf[:, 14] - self.obs_buf[:, 18])) * (2*self.action_change_penalty)
        smooth_penalty = torch.sum(torch.square((self.obs_buf[:, 11:14] - 2*self.obs_buf[:, 15:18] + self.prev_prev_actions[:, 0:3])), dim = -1) * self.smooth_penalty
        smooth_penalty += torch.sum(torch.square((self.obs_buf[:, 14] - 2*self.obs_buf[:, 18] + self.prev_prev_actions[:, 3]))) * (2*self.smooth_penalty)
        
        # lin_vel_reward = self.lin_strength * torch.linalg.norm(self.privilege_obs_buf[:, 3:5], dim=1) 
        # lin_vel_reward = self.lin_strength * torch.linalg.norm(self.obs_buf[:, 19:21], dim=1) 
        lin_vel_reward = self.lin_strength * torch.sum(torch.square(self.obs_buf[:, 19:21]), dim=-1) 

        # velo_rate_penalty = torch.sum(torch.square(self.prev_lin_vel - self.privilege_obs_buf[:, 3:6]), dim=-1) * self.lin_vel_rate_penalty
        velo_rate_penalty = torch.sum(torch.square(self.prev_lin_vel - self.obs_buf[:, 19:22]), dim=-1) * self.lin_vel_rate_penalty
        # pose_penalty = torch.sum(torch.square(self.obs_buf[:, 3:7] - self.start_rotation), dim=-1) * self.pose_penalty
        pose_penalty = torch.sum(torch.abs(self.obs_buf[:, 3:7] - self.start_rotation), dim=-1) * self.pose_penalty

        # height_penalty = torch.abs(self.obs_buf[:,0] - self.height_target) * self.height_penalty
        height_penalty = torch.square(self.obs_buf[:,0] - self.height_target) * self.height_penalty
        height_change_penalty = self.height_change_penalty * torch.square(self.obs_buf[:,1])
        # jitter_penalty = self.jitter_penalty * torch.square(self.obs_buf[:,2])
        jitter_penalty = self.jitter_penalty * torch.square(self.privilege_obs_buf[:,8])
        out_map_penalty = self.out_map_penalty * (
            torch.clamp(self.map_x_min - self.privilege_obs_buf[:, 0], min=0) ** 2 +
            torch.clamp(self.privilege_obs_buf[:, 0] - self.map_x_max, min=0) ** 2 +
            torch.clamp(self.map_y_min - self.privilege_obs_buf[:, 1], min=0) ** 2 +
            torch.clamp(self.privilege_obs_buf[:, 1] - self.map_y_max, min=0) ** 2
        )
        map_center_penalty = self.map_center_penalty * torch.square(self.privilege_obs_buf[:,2])

        # Reward getting close to waypoints
        wp_reward = torch.zeros(self.num_envs, device=self.device)
        for i, waypoint in enumerate(self.reward_wp):
            waypoint = waypoint.repeat((self.num_envs,1))
            distances = torch.norm(self.privilege_obs_buf[:, 0:3] - waypoint, dim=1)**2
            factor = (torch.exp(1 / (distances+torch.ones(self.num_envs, device=self.device))) - torch.ones(self.num_envs, device=self.device)) # 0-1
            # factor = torch.exp(-distances) # 0-1
            wp_reward += factor * self.waypoint_strength

        # Reward tracking reference trajectory
        ref_x = self.privilege_obs_buf[:, 0] + self.point_could_origin_offset[0, 0]  # Shape: [num_envs]
        ref_traj_x = self.ref_traj[:, 0]  # Shape: [ref_traj_length]
        diff = ref_traj_x.unsqueeze(0) - ref_x.unsqueeze(1)  # Shape: [num_envs, ref_traj_length]
        mask = diff > 0.5  # Shape: [num_envs, ref_traj_length]
        diff_masked = torch.where(mask, diff, float('inf'))  # Shape: [num_envs, ref_traj_length]
        min_diffs, min_indices = torch.min(diff_masked, dim=1)  # Shapes: [num_envs], [num_envs]
        no_valid_indices = torch.isinf(min_diffs)  # Shape: [num_envs]
        target_list = torch.empty((self.num_envs, 3), device=self.device)
        default_target = self.target[0] + self.point_could_origin_offset[0]  # Shape: [3]
        target_list[:] = default_target
        valid_indices = ~no_valid_indices
        selected_indices = min_indices[valid_indices]
        targets = self.ref_traj[selected_indices]  # Shape: [num_valid_envs, 3]
        target_list[valid_indices] = targets
        target_list = target_list - self.point_could_origin_offset

        desire_velo_norm = (target_list - self.privilege_obs_buf[:, 0:3]) / torch.clamp(torch.norm(target_list - self.privilege_obs_buf[:, 0:3], dim=1, keepdim=True), min=1e-6)
        curr_velo_norm = self.obs_buf[:, 19:22] / torch.clamp(torch.norm(self.obs_buf[:, 19:22], dim=1, keepdim=True), min=1e-2)
        # curr_velo_norm = self.privilege_obs_buf[:, 3:6] / torch.clamp(torch.norm(self.privilege_obs_buf[:, 3:6], dim=1, keepdim=True), min=1e-2)
        # velo_dist = torch.linalg.norm(curr_velo_norm[:, [0,2]] - desire_velo_norm[:, [0,2]], dim=1)
        velo_dist = torch.linalg.norm(curr_velo_norm - desire_velo_norm, dim=1)
        target_reward = velo_dist * self.target_factor

        # Obstacle avoidance reward
        obst_reward = torch.tensor([0.], requires_grad=True, device=self.device)
        self.time_report.start_timer("point cloud collision check")
        obst_dist = self.point_cloud.compute_nearest_distances(drone_pos, drone_rot)
        self.time_report.end_timer("point cloud collision check")
        # reward larger distance away from obstacles
        obst_reward = obst_reward + torch.where(obst_dist < self.obst_threshold, obst_dist*self.obstacle_strength, torch.zeros_like(obst_dist))
        # penalty collision
        obst_reward = obst_reward + torch.where(obst_dist < self.obst_collision_limit, torch.ones_like(obst_dist)*self.collision_penalty, torch.zeros_like(obst_dist))
        

        self.rew_buf = (
                        survive_reward + 
                        obst_reward + 
                        # ang_vel_penalty + 
                        # ang_acc_penalty +
                        pose_penalty +
                        lin_vel_reward + 
                        velo_rate_penalty +
                        up_reward + 
                        heading_reward + 
                        target_reward + 
                        action_penalty + 
                        action_change_penalty +
                        smooth_penalty +
                        height_penalty + 
                        out_map_penalty + 
                        map_center_penalty +
                        height_change_penalty +
                        jitter_penalty +
                        wp_reward 
                        )
            
        
        # reset agents
        condition_dist = (obst_dist < self.obst_collision_limit)
        condition_loss = (self.rew_buf < -10)
        # condition_senity = ~(((self.obs_buf > -1000) & (self.obs_buf < 1000) & ~torch.isnan(self.obs_buf)& ~torch.isinf(self.obs_buf)).all(dim=1))
        condition_senity = ~(( ~torch.isnan(self.obs_buf)& ~torch.isinf(self.obs_buf)).all(dim=1))
        condition_body_rate = torch.linalg.norm(self.privilege_obs_buf[:,10:13], dim=1) > self.body_rate_threshold
        condition_height = (self.privilege_obs_buf[:,2] > 4.) | (self.privilege_obs_buf[:,2] < 0.)
        condition_out_of_bounds = (
            (self.privilege_obs_buf[:, 0] < self.map_limit_fact * self.map_x_min) |
            (self.privilege_obs_buf[:, 0] > self.map_limit_fact * self.map_x_max) |
            (self.privilege_obs_buf[:, 1] < self.map_limit_fact * self.map_y_min) |
            (self.privilege_obs_buf[:, 1] > self.map_limit_fact * self.map_y_max)
        )
        condition_train = self.progress_buf > self.episode_length - 1
        combined_condition = condition_body_rate | condition_train
        if self.early_termination:
            if self.condition_approach == 1:
                combined_condition = combined_condition | condition_senity | condition_loss
            elif self.condition_approach == 2:
                combined_condition = combined_condition | condition_height | condition_senity | condition_loss
            elif self.condition_approach == 3:
                combined_condition = condition_dist | condition_height | combined_condition
            elif self.condition_approach == 4:
                combined_condition = condition_height | combined_condition | condition_out_of_bounds | condition_senity
        
        # ## used for adding breakpoint
        # if self.episode_count % 20 == 0:
        # if self.episode_count > 140:
        #     print(self.episode_count)

        self.reset_buf = torch.where(combined_condition, torch.ones_like(self.reset_buf), self.reset_buf)

        if self.visualize:
            if self.episode_count == (self.episode_length-1) or combined_condition.any():
                self.save_recordings()

    def calculateReward_new(self):
        torso_pos = self.privilege_obs_buf[:, 0:3]
        # torso_rot = self.obs_buf[:, 3:7]
        torso_quat = self.obs_buf[:, 3:7] # (x,y,z,w)
        drone_rot = torso_quat # (x, y, z, w)
        drone_pos = torso_pos + self.point_could_origin_offset # in point cloud dimension
        drone_target = self.target + self.point_could_origin_offset

        # # make yaw angle align with velocity
        # target_dirs = tu.normalize(self.privilege_obs_buf[:, [3,5]])
        # rpy = quaternion_to_euler(torso_quat[:, [3,0,2,1]]) # input is (w,x,z,y)
        # heading_vec = torch.cat([torch.cos(rpy[:,2].unsqueeze(-1)), torch.sin(rpy[:,2].unsqueeze(-1))], dim=1) 
        # yaw_alignment = (heading_vec * target_dirs).sum(dim = -1)

        # Compute ref_x for all environments
        ref_x = self.privilege_obs_buf[:, 0] + self.point_could_origin_offset[0, 0]  # Shape: [num_envs]
        ref_traj_x = self.ref_traj[:, 0]  # Shape: [ref_traj_length]
        diff = ref_traj_x.unsqueeze(0) - ref_x.unsqueeze(1)  # Shape: [num_envs, ref_traj_length]
        mask = diff > 0.5  # Shape: [num_envs, ref_traj_length]
        diff_masked = torch.where(mask, diff, float('inf'))  # Shape: [num_envs, ref_traj_length]
        min_diffs, min_indices = torch.min(diff_masked, dim=1)  # Shapes: [num_envs], [num_envs]
        no_valid_indices = torch.isinf(min_diffs)  # Shape: [num_envs]
        target_list = torch.empty((self.num_envs, 3), device=self.device)
        default_target = self.target[0] + self.point_could_origin_offset[0]  # Shape: [3]
        target_list[:] = default_target
        valid_indices = ~no_valid_indices
        selected_indices = min_indices[valid_indices]
        targets = self.ref_traj[selected_indices]  # Shape: [num_valid_envs, 3]
        target_list[valid_indices] = targets
        target_list = target_list - self.point_could_origin_offset
        obst_dist = self.point_cloud.compute_nearest_distances(drone_pos, drone_rot)

        desire_velo_norm = (target_list - self.privilege_obs_buf[:, 0:3]) / torch.clamp(torch.norm(target_list - self.privilege_obs_buf[:, 0:3], dim=1, keepdim=True), min=1e-6)
        curr_velo_norm = self.obs_buf[:, 19:22] / torch.clamp(torch.norm(self.obs_buf[:, 19:22], dim=1, keepdim=True), min=1e-2)
        # velo_dist = torch.linalg.norm(curr_velo_norm[:, [0,2]] - desire_velo_norm[:, [0,2]], dim=1)
        velo_dist = torch.linalg.norm(curr_velo_norm - desire_velo_norm, dim=1)


        # Converted reward components
        survive_reward = self.survive_reward
        up_reward = self.up_strength * self.obs_buf[:, 10]
        # heading_reward = self.heading_strength * yaw_alignment
        
        # Converted linear velocity reward (reward for target-aligned movement)
        desired_vel = self.privilege_obs_buf[:, 3:5]
        actual_vel = self.obs_buf[:, 19:21]
        vel_alignment = (desired_vel * actual_vel).sum(dim=1)
        lin_vel_reward = self.lin_strength * vel_alignment
        
        # Action rewards (reward for smaller/consistent actions)
        action_reward = self.action_reward * (1.0 / (1.0 + torch.sum(torch.square(self.obs_buf[:, 11:15]), dim=-1)))
        action_consistency = 1.0 / (1.0 + torch.sum(torch.square(self.obs_buf[:, 11:15] - self.obs_buf[:, 15:19]), dim=-1))
        action_consistency_reward = self.action_consistency_reward * action_consistency
        
        # Height rewards
        height_diff = torch.abs(self.obs_buf[:,0] - self.height_target)
        height_reward = self.height_reward * (1.0 / (1.0 + height_diff))
        height_stability = 1.0 / (1.0 + torch.square(self.obs_buf[:,1]))
        height_stability_reward = self.height_stability_reward * height_stability
        
        # Smoothness reward
        smoothness = 1.0 / (1.0 + torch.square(self.privilege_obs_buf[:,8]))
        smoothness_reward = self.smoothness_reward * smoothness
        
        # Map boundary rewards
        in_bounds = torch.ones_like(self.privilege_obs_buf[:,0])
        in_bounds = in_bounds - torch.clamp(self.map_x_min - self.privilege_obs_buf[:, 0], min=0)
        in_bounds = in_bounds - torch.clamp(self.privilege_obs_buf[:, 0] - self.map_x_max, min=0)
        in_bounds = in_bounds - torch.clamp(self.map_y_min - self.privilege_obs_buf[:, 1], min=0)
        in_bounds = in_bounds - torch.clamp(self.privilege_obs_buf[:, 1] - self.map_y_max, min=0)
        in_bounds_reward = self.in_bounds_reward * in_bounds
        
        center_distance = torch.norm(self.privilege_obs_buf[:, 0:2], dim=1)
        map_center_reward = self.map_center_reward * (1.0 / (1.0 + center_distance))
        
        # Converted waypoint following
        wp_reward = torch.zeros(self.num_envs, device=self.device)
        for i, waypoint in enumerate(self.reward_wp):
            distances = torch.norm(self.privilege_obs_buf[:, 0:3] - waypoint, dim=1)
            wp_reward += self.waypoint_strength * (1.0 / (1.0 + distances))

        # Target alignment reward
        vel_alignment = (curr_velo_norm * desire_velo_norm).sum(dim=1)
        target_reward = self.target_alignment_factor * vel_alignment

        # Obstacle rewards
        obst_reward = self.obstacle_clearance_reward * (obst_dist / self.obst_threshold)
        collision_penalty = torch.where(obst_dist < self.obst_collision_limit, 
                                    self.collision_penalty, 
                                    torch.zeros_like(obst_dist))
        self.rew_buf = (
            survive_reward +
            obst_reward +
            lin_vel_reward +
            up_reward +
            # heading_reward +
            target_reward +
            action_reward +
            action_consistency_reward +
            height_reward +
            height_stability_reward +
            smoothness_reward +
            in_bounds_reward +
            map_center_reward +
            wp_reward +
            collision_penalty  # Only remaining penalty
        )

        # reset agents
        condition_dist = (obst_dist < self.obst_collision_limit)
        condition_loss = (self.rew_buf < -10)
        # condition_senity = ~(((self.obs_buf > -1000) & (self.obs_buf < 1000) & ~torch.isnan(self.obs_buf)& ~torch.isinf(self.obs_buf)).all(dim=1))
        condition_senity = ~(( ~torch.isnan(self.obs_buf)& ~torch.isinf(self.obs_buf)).all(dim=1))
        condition_body_rate = torch.linalg.norm(self.privilege_obs_buf[:,10:13], dim=1) > self.body_rate_threshold
        condition_height = (self.privilege_obs_buf[:,2] > 4.) | (self.privilege_obs_buf[:,2] < 0.)
        condition_out_of_bounds = (
            (self.privilege_obs_buf[:, 0] < self.map_limit_fact * self.map_x_min) |
            (self.privilege_obs_buf[:, 0] > self.map_limit_fact * self.map_x_max) |
            (self.privilege_obs_buf[:, 1] < self.map_limit_fact * self.map_y_min) |
            (self.privilege_obs_buf[:, 1] > self.map_limit_fact * self.map_y_max)
        )
        condition_train = self.progress_buf > self.episode_length - 1
        combined_condition = condition_body_rate | condition_train
        if self.early_termination:
            if self.condition_approach == 1:
                combined_condition = combined_condition | condition_senity | condition_loss
            elif self.condition_approach == 2:
                combined_condition = combined_condition | condition_height | condition_senity | condition_loss
            elif self.condition_approach == 3:
                combined_condition = condition_dist | condition_height | combined_condition
            elif self.condition_approach == 4:
                combined_condition = condition_height | combined_condition | condition_out_of_bounds | condition_senity
        
        # ## used for adding breakpoint
        # if self.episode_count % 20 == 0:
        # if self.episode_count > 140:
        #     print(self.episode_count)

        self.reset_buf = torch.where(combined_condition, torch.ones_like(self.reset_buf), self.reset_buf)

        if self.visualize:
            if self.episode_count == (self.episode_length-1) or combined_condition.any():
                self.save_recordings()


    def save_recordings(self):
        save_path = self.save_path
        np.save(f'{save_path}/x.npy', np.array(self.x_record, dtype=object), allow_pickle=True)
        np.save(f'{save_path}/y.npy', np.array(self.y_record, dtype=object), allow_pickle=True)
        np.save(f'{save_path}/z.npy', np.array(self.z_record, dtype=object), allow_pickle=True)

        # # figure x over time
        # plt.figure()
        # plt.plot(range(len(self.x_record)), self.x_record)
        # plt.xlabel("Step")
        # plt.ylabel("X/m")
        # plt.savefig(f'{save_path}/x_plot.png')

        # # figure y over time
        # plt.figure()
        # plt.plot(range(len(self.y_record)), self.y_record)
        # plt.xlabel("Step")
        # plt.ylabel("Y/m")
        # plt.savefig(f'{save_path}/y_plot.png')

        # # figure z over time
        plt.figure()
        plt.plot(range(len(self.z_record)), self.z_record)
        plt.xlabel("Step")
        plt.ylabel("Z/m")
        plt.savefig(f'{save_path}/z_plot.png')

        # figure pose over time
        plt.figure()
        plt.plot(range(len(self.x_record)), self.x_record)
        plt.plot(range(len(self.y_record)), self.y_record)
        plt.plot(range(len(self.z_record)), self.z_record)
        plt.xlabel("Step")
        plt.ylabel("m")
        plt.legend(['x', 'y', 'z'])
        plt.savefig(f'{save_path}/pos_plot.png')

        # figure traj 
        wp_np = self.reward_wp.clone().detach().cpu().numpy()
        target = self.target[0,:].clone().detach().cpu().numpy()
        plt.figure()
        plt.plot(self.x_record, self.y_record)

        # draw waypoints
        wp_num, _ = wp_np.shape
        if wp_num > 0:
            for i in range(wp_num-1):
                circle = plt.Circle((wp_np[i][0], wp_np[i][1]), 0.3, color='b', fill=True)
                plt.gca().add_patch(circle)
            circle = plt.Circle((wp_np[-1][0], wp_np[-1][1]), 0.5, color='r', fill=True)
            plt.gca().add_patch(circle)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("X/m")
        plt.ylabel("Y/m")
        plt.savefig(f'{save_path}/traj_plot.png')

        # figure pose over time
        plt.figure()
        plt.plot(range(len(self.roll_record)), self.roll_record)
        plt.plot(range(len(self.pitch_record)), self.pitch_record)
        plt.plot(range(len(self.yaw_record)), self.yaw_record)
        plt.xlabel("Step")
        plt.ylabel("rad")
        plt.legend(['r', 'p', 'y'])
        plt.savefig(f'{save_path}/pose_plot.png')

        # figure velo over time
        plt.figure()
        plt.plot(range(len(self.velo_x_record)), self.velo_x_record)
        plt.plot(range(len(self.velo_y_record)), self.velo_y_record)
        plt.plot(range(len(self.velo_z_record)), self.velo_z_record)
        plt.plot(range(len(self.vae_velo_x_record)), self.vae_velo_x_record, linestyle='dashed')
        plt.plot(range(len(self.vae_velo_y_record)), self.vae_velo_y_record, linestyle='dashed')
        plt.plot(range(len(self.vae_velo_z_record)), self.vae_velo_z_record, linestyle='dashed')
        plt.xlabel("Step")
        plt.ylabel("m/s")
        plt.legend(['vx', 'vy', 'vz', 'vae_vx', 'vae_vy', 'vae_vz'])
        plt.savefig(f'{save_path}/velo_plot.png')


        # figure depth 
        plt.figure()
        plt.plot(range(len(self.depth_record)), self.depth_record)
        plt.xlabel("Step")
        plt.ylabel("depth/m")
        plt.savefig(f'{save_path}/depth_plot.png')

        # figure action
        plt.figure()
        for action_id in range(3):
            plt.plot(range(np.shape(self.action_record)[0]), self.action_record[:,action_id], label=f'rotor {action_id}') 
        plt.plot(range(len(self.ang_velo_x_record)), self.ang_velo_x_record, linestyle='dashed')
        plt.plot(range(len(self.ang_velo_y_record)), self.ang_velo_y_record, linestyle='dashed')
        plt.plot(range(len(self.ang_velo_z_record)), self.ang_velo_z_record, linestyle='dashed')
        plt.xlabel('Step')
        plt.ylabel('body_rate')
        plt.legend(['r_des', 'p_des', 'y_des', 'r', 'p', 'y'])
        plt.savefig(f'{save_path}/body_rate_plot.png')
        np.savetxt(f'{save_path}/action.txt', self.action_record, delimiter=',')


        plt.figure()
        plt.plot(range(np.shape(self.action_record)[0]), self.action_record[:,3]) 
        plt.xlabel('Step')
        plt.ylabel('normalized_force')
        plt.savefig(f'{save_path}/action_force_plot.png')

        # save video
        video_tensor = torch.permute(torch.stack(self.img_record), (0, 2, 3, 1)) * 255
        video_tensor = video_tensor.to('cpu')
        video_tensor_uint8 = video_tensor.to(dtype=torch.uint8)
        write_video(f'{save_path}/ego_video.mp4', video_tensor_uint8, fps=30)

    
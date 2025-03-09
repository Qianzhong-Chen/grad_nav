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

SINGLE_VISUAL_INPUT_SIZE = 16
HISTORY_BUFFER_NUM = 5
LATENT_VECT_NUM = 24
DEPTH_DATA = False

class VisualPerceptionNet(nn.Module):
    def __init__(self, input_channels=3):
        super(VisualPerceptionNet, self).__init__()
        if DEPTH_DATA:
            input_channels = 4
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


class DroneMultiGateEnv(DFlexEnv):

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
        num_obs = 17 + SINGLE_VISUAL_INPUT_SIZE + LATENT_VECT_NUM # correspond with self.obs_buf
        num_act = 4
        self.agent_name = 'drone_multi_gate'
        self.num_history = HISTORY_BUFFER_NUM
        self.num_latent = LATENT_VECT_NUM
    
        print('----------------------------', device)
        super(DroneMultiGateEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.device = device
        self.time_report = TimeReport()
        self.time_report.add_timer("dynamic simulation")
        self.time_report.add_timer("3D GS inference")
        self.time_report.add_timer("point cloud collision check")

        # setup map
        self.map_lib = {
            "gate_left":"sv_917_3_left_nerfstudio",
            "gate_right":"sv_917_3_right_nerfstudio",
            "simple_hover":"sv_917_3_right_nerfstudio",
            "gate_mid":"sv_1007_gate_mid",
            "clutter":"sv_712_nerfstudio",
            "backroom":"sv_1018_2",
            "flightroom":"sv_1018_3",
        }
        self.map_name = map_name
        self.multi_map = ["gate_right", "gate_left", "gate_mid"]
        self.multi_map_num = len(self.multi_map)
        

        # SysID value
        self.min_mass = 1.0
        self.mass_range = 0.2
        self.min_thrust = 24.0
        self.thrust_range = 4.0
        self.obs_noise_level = 0.05
        self.br_delay_factor = 0.8
        self.thrust_delay_factor = 0.7
        self.start_height = 1.35
        self.target_height = 1.40
        self.init_inertia = [0.01, 0.012, 0.025]
        self.init_kp = [1.0, 1.2, 2.5]
        self.init_kd = [0.001, 0.001, 0.002]

        self.init_sim()
        self.episode_length = episode_length

        # GS model
        gs_dir = Path("assets/gs_data")
        self.resolution_quality = 0.8
        self.gs = get_gs(self.map_name, gs_dir, self.resolution_quality)
        

        # navigation parameters
        if self.map_name == 'gate_mid':
            target = (13.0, -2.0, 1.3) # in map absolute dimension, start with 0
            target_xy = (13.0, -2.0)
        if self.map_name == 'gate_left':
            target = (13.0, -2.0, 1.3) # in map absolute dimension, start with 0
            target_xy = (13.0, -2.0)
        if self.map_name == 'gate_right':
            target = (13.0, -2.0, 1.3) # in map absolute dimension, start with 0
            target_xy = (13.0, -2.0)
        
       
        self.target = tu.to_torch(list(target), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # navigation target
        self.target_xy = tu.to_torch(list(target_xy), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # navigation target
        # self.target_list = self.target.clone()
        self.gs_origin_offset = torch.tensor([[-6.0, -0., -0.05]] * self.num_envs, device=self.device) # (x,y,z); gs dimension for visual info
        self.point_could_origin_offset = torch.tensor([[-6.0, -0., -0.05]] * self.num_envs, device=self.device) # (x,y,z); point cloud dimension for obstacle distance & traj plan
        # Initialize the reference trajectory for each map
        self.init_multi_map()
        
        # setup point cloud
        script_dir = Path(__file__).parent
        point_cloud_dir = script_dir / "assets" / "point_cloud"
        ply_file = point_cloud_dir / f"{self.map_lib[self.map_name]}.ply"
        self.point_cloud = ObstacleDistanceCalculator(ply_file=ply_file)
        self.reward_wp = self.reward_wp_table[self.map_name]
        self.ref_traj = self.ref_traj_table[self.map_name]
        
        
        self.reward_wp_list = [self.reward_wp + self.point_could_origin_offset[0].repeat((self.reward_wp.shape[0],1))] * self.num_envs
        self.reward_wp_record = []
        for i in range(self.reward_wp.shape[0]):
            self.reward_wp_record.append(torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        
        # Reward function
        self.up_strength = 0.25
        self.heading_strength = 0.1 # reward yaw motion align with lin_vel
        self.lin_strength = -2.0
        self.yaw_strength = 0.0
        self.action_penalty = -1.0
        self.action_change_penalty = -1.0
        self.smooth_penalty = -1.0
        # self.ang_vel_penalty = -0.4
        # self.ang_acc_penalty = -0.005
        self.survive_reward = 8.0
        self.pose_penalty = -0.5
        self.height_penalty = -3.0
        self.height_change_penalty = -0.5
        self.jitter_penalty = -0.1
        self.out_map_penalty = -1.5
        self.map_center_penalty = -0.05
        # trajectory
        self.waypoint_strength = 5.0
        self.target_factor = -5.0
        # obstacle avoidance
        self.obstacle_strength = 0.75
        self.collision_penalty_coef = 0.0
        self.collision_penalty = self.collision_penalty_coef*self.survive_reward 
        self.obst_threshold = 0.5
        self.obst_collision_limit = 0.20
        self.body_rate_threshold = 15
        self.domain_randomization = False

        

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
            "up_strength": self.up_strength,
            "heading_strength": self.heading_strength,
            "lin_strength": self.lin_strength,
            "obstacle_strength": self.obstacle_strength,
            "obst_collision_limit": self.obst_collision_limit,
            "action_penalty": self.action_penalty,
            "smooth_penalty": self.smooth_penalty,
            "target_factor": self.target_factor, 
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
            "action_change_penalty": self.action_change_penalty,
            "min_mass": self.min_mass,
            "mass_range": self.mass_range,
            "min_thrust": self.min_thrust,
            "min_thrust": self.thrust_range,
            "resolution_quality": self.resolution_quality,
            "br_delay_factor": self.br_delay_factor,
            "thrust_delay_factor":self.thrust_delay_factor,
            "start_height": self.start_height,
            "target_height": self.target_height,
        }

        
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
        
    def init_multi_map(self, ):
        self.ref_traj_table = {}
        self.reward_wp_table = {}

        # setup point cloud
        point_cloud_dir = Path("assets/point_cloud")
        
        # generate ref_traj
        for map_name in self.multi_map:
            if map_name == 'gate_mid':
                reward_wp = torch.tensor([
                                        [4.0, 0.0, 1.35],
                                        [5.8, -0.1, 1.45], 
                                        ], device=self.device) # (x,y,z)
                traj_start = torch.tensor([[-6.0, 0, 1.3]], device=self.device)
                traj_dest = torch.tensor([[2.0, 0.0, 1.3]], device=self.device)
            elif map_name == 'gate_left':
                reward_wp = torch.tensor([
                                        [4.0, 0.9, 1.35],    
                                        [6.1, 1.1, 1.45], 
                                        ], device=self.device) # (x,y,z)
                traj_start = torch.tensor([[-6.0, 0, 1.3]], device=self.device)
                traj_dest = torch.tensor([[2.0, 1.0, 1.3]], device=self.device)
            elif map_name == 'gate_right':
                reward_wp = torch.tensor([
                                        [4.0, -1.3, 1.35],  
                                        [6.1, -1.2, 1.45], 
                                        ], device=self.device) # (x,y,z)
                traj_start = torch.tensor([[-6.0, 0, 1.3]], device=self.device)
                traj_dest = torch.tensor([[2.0, -1.0, 1.3]], device=self.device)
            
                
            # plane the global ref trajectory
            traj_planner = TrajectoryPlanner(ply_file=os.path.join(point_cloud_dir,f'{self.map_lib[map_name]}.ply'), 
                                            safety_distance=0.15, 
                                            batch_size=1, 
                                            wp_distance=2.0,
                                            verbose=False)
            traj_wp = reward_wp + self.point_could_origin_offset[0].repeat((reward_wp.shape[0],1))
            ref_traj = traj_planner.plan_trajectories(traj_start, traj_dest, [traj_wp])
            ref_traj = torch.tensor(ref_traj[0], device = self.device)
            self.ref_traj_table[map_name] = ref_traj
            self.reward_wp_table[map_name] = reward_wp


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

        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # initialize some data used later on
        # todo - switch to z-up
        self.up_vec = self.z_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.mass = torch.full((self.num_envs,), self.min_mass+0.5*self.mass_range, device=self.device)  # All drones have mass = 1.2 kg
        self.max_thrust = torch.full((self.num_envs,), self.min_thrust+0.5*self.thrust_range, device=self.device)  # All drones have max_thrust = 23 N
        self.hover_thrust = self.mass * 9.81  # Hover thrust for each drone
        inertia = torch.tensor(self.init_inertia, device=self.device)  # Diagonal inertia components
        self.inertia = torch.diag(inertia).unsqueeze(0).repeat(self.num_envs, 1, 1)  # Repeat for all drones
        link_length = 0.15  # meters
        Kp = torch.tensor(self.init_kp, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # Repeat for all drones
        Kd = torch.tensor(self.init_kd, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)  # Repeat for all drones

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
       
        self.start_pos = []
        # self.start_body_rate = [0., 0., 0.] # r,p,y
        self.start_norm_thrust = [(self.hover_thrust / self.max_thrust).clone().detach().cpu().numpy()[0]]
        self.control_base = self.start_norm_thrust[0]
        self.start_action = self.start_body_rate + self.start_norm_thrust
        

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

    def change_map(self, map_name):
        assert map_name in list(self.map_lib.keys())
        self.map_name = map_name

        # Load 3D GS
        gs_dir = Path("assets/gs_data")
        self.gs = get_gs(self.map_name, gs_dir, self.resolution_quality)

        self.reward_wp = self.reward_wp_table[self.map_name]
        self.ref_traj = self.ref_traj_table[self.map_name]
        
    
        self.reward_wp_list = [self.reward_wp + self.point_could_origin_offset[0].repeat((self.reward_wp.shape[0],1))] * self.num_envs
        self.reward_wp_record = []
        for i in range(self.reward_wp.shape[0]):
            self.reward_wp_record.append(torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        
    # Forward simulation of the trajectory
    def step(self, actions, vae_info, vel_net_info):
        self.latent_vect, _ = vae_info[0].clone().detach(), vae_info[1].clone().detach()
        self.lin_vel_vae = vel_net_info[0].clone().detach()

        # prepare vae data
        self.obs_hist_buf.update(self.vae_obs_buf)
        obs_hist = self.obs_hist_buf.get_concatenated().clone().detach()
        obs_vel = self.privilege_obs_buf[:, 3:6].clone().detach()

        actions = actions.view((self.num_envs, self.num_actions))
        body_rate_cols = torch.clip(actions[:, 0:3], -1., 1.) * 0.5

        # Process actions
        prev_body_rate = self.prev_actions[:, 0:3].clone().detach()
        body_rate_cols = self.br_delay_factor*(torch.clip(actions[:, 0:3], -1., 1.) * 0.5) + (1-self.br_delay_factor)*prev_body_rate
        body_rate_cols = torch.clip(body_rate_cols, -0.5, 0.5) # limit the body rate
        prev_thrust = self.prev_actions[:, -1].unsqueeze(-1).clone().detach()
        thrust_col = self.thrust_delay_factor*((torch.clip(actions[:, 3:],-1., 1.) + 1) * 0.25) + \
            (1-self.thrust_delay_factor)*prev_thrust # rough simulation of motor delay, thrust range is [0, 0.5]

        actions = torch.cat([body_rate_cols, thrust_col], dim=1)
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()
        self.actions = actions # normalized
        control_input = (actions[:, 0:3], actions[:, 3])
        
        # Update the state variables
        torso_pos = self.state_joint_q.view(self.num_envs, -1)[:, 0:3] # (x, y, z)
        torso_quat = self.state_joint_q.view(self.num_envs, -1)[:, 3:7] # rotated 90 deg
        lin_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 3:6] # joint_qd rot has 3 entries
        ang_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 0:3]
        lin_acc = self.state_joint_qdd.view(self.num_envs, -1)[:, 3:6] # joint_qd rot has 3 entries
        ang_acc = self.state_joint_qdd.view(self.num_envs, -1)[:, 0:3]
        
        # Core dynamic simulation
        self.time_report.start_timer("dynamic simulation")
        new_position, new_linear_velocity, new_angular_velocity, new_quaternion, new_linear_acceleration, new_angular_acceleration = \
        self.quad_dynamics.run_simulation(
                                            position=torso_pos,
                                            velocity=lin_vel,
                                            orientation=torso_quat[:, [3,0,1,2]], # (w, x, y, z)
                                            angular_velocity=ang_vel,
                                            control_input=control_input
                                        )
        self.time_report.end_timer("dynamic simulation")        # set new state back
        if self.no_grad:
            new_position = new_position.detach()
            new_quaternion = new_quaternion.detach()
            new_linear_velocity = new_linear_velocity.detach()
            new_angular_velocity = new_angular_velocity.detach()

        # Update the state variables
        self.state_joint_q.view(self.num_envs, -1)[:, 0:3] = new_position
        self.state_joint_q.view(self.num_envs, -1)[:, 3:7] = new_quaternion[:, [1,2,3,0]].clone()
        self.state_joint_qd.view(self.num_envs, -1)[:, 3:6] = new_linear_velocity
        self.state_joint_qd.view(self.num_envs, -1)[:, 0:3] = new_angular_velocity
        self.state_joint_qdd.view(self.num_envs, -1)[:, 3:6] = new_linear_acceleration.clone().detach()
        self.state_joint_qdd.view(self.num_envs, -1)[:, 0:3] = new_angular_acceleration.clone().detach()
        self.sim_time += self.sim_dt
        self.reset_buf = torch.zeros_like(self.reset_buf)

        self.progress_buf += 1
        self.num_frames += 1

        self.calculateObservations() # Update obs
        self.calculateReward() # Update reward

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

        # Senity check
        if (torch.isnan(obs_hist).any() | torch.isinf(obs_hist).any()):
            print('obs hist nan')
            obs_hist = torch.nan_to_num(obs_hist, nan=0.0, posinf=1e3, neginf=-1e3)
        if (torch.isnan(obs_vel).any() | torch.isinf(obs_vel).any()):
            print('obs vel nan')
            obs_vel = torch.nan_to_num(obs_vel, nan=0.0, posinf=1e3, neginf=-1e3)

        return self.obs_buf, self.privilege_obs_buf, obs_hist, obs_vel, self.rew_buf, self.reset_buf, self.extras
    
    

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
                inertia = torch.tensor(self.init_inertia, device=self.device)  # Base inertia
                inertial_noise = (torch.rand(num_reset_envs, 3, device=self.device) - 0.5) * 2 * 0.2 * inertia  # ±20% noise
                randomized_inertia = inertia + inertial_noise
                self.inertia[env_ids] = torch.diag_embed(randomized_inertia)  # Shape: (num_reset_envs, 3, 3)
                self.hover_thrust[env_ids] = self.mass[env_ids] * 9.81

                # Reinitialize QuadrotorSimulator with updated parameters
                self.quad_dynamics = QuadrotorSimulator(
                    mass=self.mass,  # Shape: (batch_size,)
                    inertia=self.inertia,  # Shape: (batch_size, 3, 3)
                    link_length=0.15,  # Scalar
                    Kp=torch.tensor(self.init_kp, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),  # Shape: (batch_size, 3)
                    Kd=torch.tensor(self.init_kd, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),  # Shape: (batch_size, 3)
                    freq=200.0,  # Scalar
                    max_thrust=self.max_thrust,  # Shape: (batch_size,)
                    total_time=self.sim_dt,  # Scalar
                    rotor_noise_std=0.01,  # Scalar
                    br_noise_std=0.01  # Scalar
                )

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
                self.state_joint_qd.view(self.num_envs, -1)[env_ids, :] = 0.5 * (torch.rand(size=(len(env_ids), 6), device=self.device) - 0.5)
                self.state_joint_qdd.view(self.num_envs, -1)[env_ids, :] = 0.05 * (torch.rand(size=(len(env_ids), 6), device=self.device) - 0.5)
                self.actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone() + 0.05 * (torch.rand(size=(len(env_ids), 4), device=self.device))
                self.prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()
                self.prev_prev_actions.view(self.num_envs, -1)[env_ids, :] = self.start_joint_q.clone()


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

        return checkpoint
    
    def process_GS_data(self, depth_list, nerf_img):
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
        
        if DEPTH_DATA:
            # Concatenate RGB image and depth data along the channel dimension
            combined_tensor = torch.cat([input_tensor, depth_tensor], dim=1)
            # Resize to match SqueezeNet input size
            resize = nn.AdaptiveAvgPool2d((224, 224))
            combined_tensor = resize(combined_tensor)
            combined_tensor = combined_tensor.to(self.device)
            self.visual_info = self.visual_net(combined_tensor)
            self.visual_info = self.visual_info.detach()
        else:
            # Only use RGB
            resize = nn.AdaptiveAvgPool2d((224, 224))
            input_tensor = resize(input_tensor)
            input_tensor = input_tensor.to(self.device)
            self.visual_info = self.visual_net(input_tensor)
            self.visual_info = self.visual_info.detach()



    def calculateObservations(self):
        torso_pos = self.state_joint_q.view(self.num_envs, -1)[:, 0:3]
        torso_quat = self.state_joint_q.view(self.num_envs, -1)[:, 3:7] # (x,y,z,w)
        lin_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 3:6] # joint_qd rot has 3 entries
        ang_vel = self.state_joint_qd.view(self.num_envs, -1)[:, 0:3]
        lin_acceleration = self.state_joint_qdd.view(self.num_envs, -1)[:, 3:6]
        ang_acceleration = self.state_joint_qdd.view(self.num_envs, -1)[:, 0:3]

        # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
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

        # Abalation variables        
        latent_abalation = torch.zeros_like(self.latent_vect)
        torso_rot_ablation = torch.zeros_like(torso_quat)
        ang_vel_ablation = torch.zeros_like(ang_vel)
        lin_vel_ablation = torch.zeros_like(lin_vel)
        torso_pos_ablation = torch.zeros_like(torso_pos)

        # adding noise
        torso_pos_noise = self.obs_noise_level * (torch.rand_like(torso_pos)-0.5)
        lin_vel_noise = self.obs_noise_level * (torch.rand_like(lin_vel) - 0.5)
        visual_noise = self.obs_noise_level * (torch.rand_like(self.visual_info) - 0.5)
        latent_noise = self.obs_noise_level * (torch.rand_like(self.latent_vect) - 0.5)
        torso_quat_noise = self.obs_noise_level * (torch.rand_like(torso_quat) - 0.5)

        self.privilege_obs_buf = torch.cat([
                                torso_pos[:, :], # 0:3 
                                # lin_vel_new, # 3:6
                                lin_vel,
                                lin_acceleration, # acc 6:9 
                                torso_quat, # 9:13
                                ang_vel, # 13:16
                                up_vec[:, 1:2], # 16
                                (heading_vec * target_dirs).sum(dim = -1).unsqueeze(-1), # 17
                                self.actions, # 18:22
                                self.prev_actions, # 22:26
                                self.depth_list, # 26
                                self.visual_info, # 27:51
                                self.latent_vect, # 51:67
                                self.lin_vel_vae,
                                ], 
                                dim = -1)

        self.obs_buf = torch.cat([
                                (torso_pos[:, 2]+torso_pos_noise[:, 2]).unsqueeze(-1), # z-pos 0
                                (lin_vel[:, 2]+lin_vel_noise[:, 2]).unsqueeze(-1), # z-vel 1
                                torso_quat+torso_quat_noise, # 2:6
                                self.actions, # 6:10
                                self.prev_actions, # 10:14
                                lin_vel, # 14:17
                                self.visual_info+visual_noise, # 17:41                                    
                                self.latent_vect+latent_noise, # 41:57
                                ], 
                                dim = -1)
            
        
        self.vae_obs_buf = torch.cat([
                                (torso_pos[:, 2]+torso_pos_noise[:, 2]).unsqueeze(-1), # z-pos 0
                                (lin_vel[:, 2]+lin_vel_noise[:, 2]).unsqueeze(-1), # z-vel 1
                                torso_quat+torso_quat_noise, # 2:6
                                self.actions, # 6:10
                                self.prev_actions, # 10:14
                                lin_vel, # 14:17
                                self.visual_info+visual_noise, # 17:41                                    
                                latent_abalation, # 41:57
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

    def calculateReward(self):
        torso_pos = self.privilege_obs_buf[:, 0:3]
        torso_quat = self.obs_buf[:, 2:6] # (x,y,z,w)
        drone_rot = torso_quat # (x, y, z, w)
        drone_pos = torso_pos + self.point_could_origin_offset # in point cloud dimension
        drone_target = self.target + self.point_could_origin_offset
       
        # use quat directly
        target_dirs = tu.normalize(self.privilege_obs_buf[:, 3:5])
        heading_vec = quaternion_yaw_forward(torso_quat)  # Extract (x, y) forward direction, input is (x, y, z, w)
        yaw_alignment = (heading_vec * target_dirs).sum(dim=-1) 


        # ## Original way
        survive_reward = self.survive_reward
        heading_reward = self.heading_strength * yaw_alignment
        
        # Control related rewards
        action_penalty = torch.sum(torch.square(self.obs_buf[:, 6:9]), dim = -1) * self.action_penalty # penalty on body rate
        action_penalty += torch.sum(torch.square(self.obs_buf[:, 9]-0.42)) * (2*self.action_penalty) # thrust penalty based on hover thrust
        action_change_penalty = torch.sum(torch.square((self.obs_buf[:, 6:9] - self.obs_buf[:, 10:13])), dim = -1) * self.action_change_penalty # penalty on action change
        action_change_penalty += torch.sum(torch.square(self.obs_buf[:, 9] - self.obs_buf[:, 13])) * (2*self.action_change_penalty)
        smooth_penalty = torch.sum(torch.square((self.obs_buf[:, 6:9] - 2*self.obs_buf[:, 10:13] + self.prev_prev_actions[:, 0:3])), dim = -1) * self.smooth_penalty
        smooth_penalty += torch.sum(torch.square((self.obs_buf[:, 9] - 2*self.obs_buf[:, 13] + self.prev_prev_actions[:, 3]))) * (2*self.smooth_penalty)

        # State related rewards
        lin_vel_reward = self.lin_strength * torch.sum(torch.square(self.obs_buf[:, 14:17]), dim=-1) 
        velo_rate_penalty = torch.sum(torch.square(self.prev_lin_vel - self.obs_buf[:, 14:17]), dim=-1) * self.lin_vel_rate_penalty
        pose_penalty = torch.sum(torch.abs(self.obs_buf[:, 2:6] - self.start_rotation), dim=-1) * self.pose_penalty
        height_penalty = torch.square(self.obs_buf[:,0] - self.target_height) * self.height_penalty
        height_change_penalty = torch.square(self.obs_buf[:,1]) * self.height_change_penalty
        jitter_penalty = torch.square(self.privilege_obs_buf[:,8]) * self.jitter_penalty 
        out_map_penalty = (
            torch.clamp(self.map_x_min - self.privilege_obs_buf[:, 0], min=0) ** 2 +
            torch.clamp(self.privilege_obs_buf[:, 0] - self.map_x_max, min=0) ** 2 +
            torch.clamp(self.map_y_min - self.privilege_obs_buf[:, 1], min=0) ** 2 +
            torch.clamp(self.privilege_obs_buf[:, 1] - self.map_y_max, min=0) ** 2
        ) * self.out_map_penalty 
        map_center_penalty = torch.square(self.privilege_obs_buf[:,2]) * self.map_center_penalty 

        # Navigation related rewards
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
        curr_velo_norm = self.obs_buf[:, 14:17] / torch.clamp(torch.norm(self.obs_buf[:, 14:17], dim=1, keepdim=True), min=1e-2)
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
                        pose_penalty +
                        lin_vel_reward + 
                        velo_rate_penalty +
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
            
        
        # Early termination settings
        # condition_dist = (obst_dist < self.obst_collision_limit)
        # condition_loss = (self.rew_buf < -10)
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
            combined_condition = condition_height | combined_condition | condition_out_of_bounds | condition_senity
        

        self.reset_buf = torch.where(combined_condition, torch.ones_like(self.reset_buf), self.reset_buf)

        if self.visualize:
            if self.episode_count == (self.episode_length-1) or combined_condition.any():
                self.save_recordings()

    


    def save_recordings(self):
        save_path = self.save_path
        np.save(f'{save_path}/x.npy', np.array(self.x_record, dtype=object), allow_pickle=True)
        np.save(f'{save_path}/y.npy', np.array(self.y_record, dtype=object), allow_pickle=True)
        np.save(f'{save_path}/z.npy', np.array(self.z_record, dtype=object), allow_pickle=True)

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
        write_video(f'{save_path}/ego_video.mp4', video_tensor_uint8, fps=20)

        # save first image
        ini_img = torch.permute(self.img_record[0], (1, 2, 0)) * 255
        ini_img = ini_img.clone().detach().cpu().numpy()
        ini_img = ini_img.astype(np.uint8)
        ini_img = Image.fromarray(ini_img)
        ini_img.save(f'{save_path}/ini_img.png')

    
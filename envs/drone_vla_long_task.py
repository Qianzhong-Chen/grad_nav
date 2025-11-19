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
import torch.nn.functional as F
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from rich.console import Console
from PIL import Image
from envs.assets.quadrotor_dynamics_advanced import QuadrotorSimulator
import numpy as np
np.set_printoptions(precision=5, linewidth=256, suppress=True)

from utils import torch_utils as tu
from utils.common import *
from utils.gs_local import GS, get_gs
from utils.rotation import quaternion_to_euler, quaternion_yaw_forward
from utils.model import ModelBuilder
from utils.point_cloud_util import ObstacleDistanceCalculator
from utils.traj_planner_vla import TrajectoryPlanner
from utils.hist_obs_buffer import ObsHistBuffer
from utils.time_report import TimeReport
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.clip import VLM
from models.squeeze_net import VisualPerceptionNet


class DroneVLALongTaskEnv(DFlexEnv):
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
                 map_name='gate_mid',
                 eval_task_id=0,
                 env_hyper=None,
                 ):

        self.agent_name = 'drone_vlm'
        self.num_history = env_hyper.get('HISTORY_BUFFER_NUM', 5)
        self.num_latent = env_hyper.get('LATENT_VECT_NUM', 24)
        self.vlm_feature_dim = env_hyper.get('VLM_FEATURE_SIZE', 256)
        self.visual_feature_size = env_hyper.get('SINGLE_VISUAL_INPUT_SIZE', 64)
        self.use_depth = env_hyper.get('DEPTH_DATA', False)
        self.num_privilege_obs = 30 + self.visual_feature_size + self.num_latent
        self.num_latent_obs = 17 + self.num_latent + self.visual_feature_size
        num_obs = 17 + self.num_latent + self.visual_feature_size # correspond with self.obs_buf
        num_act = 4
        
    
        print('----------------------------', device)
        super(DroneVLALongTaskEnv, self).__init__(num_envs, num_obs, num_act, episode_length, MM_caching_frequency, seed, no_grad, render, device)
        self.stochastic_init = stochastic_init
        self.early_termination = early_termination
        self.device = device
        self.time_report = TimeReport()
        self.time_report.add_timer("dynamic simulation")
        self.time_report.add_timer("3D GS inference")
        self.time_report.add_timer("point cloud collision check")
        self.time_report.add_timer("VLM inference")

        # setup map
        maps = {
            "gate_mid_new": "gate_mid_new",
        }
        self.map_name = map_name

        # SysID value
        self.min_mass = 1.0
        self.mass_range = 0.2
        self.min_thrust = 24.0
        self.thrust_range = 4.0
        self.obs_noise_level = 0.1
        self.br_delay_factor = 0.8
        self.thrust_delay_factor = 0.7
        self.start_height = 1.25
        self.target_height = 1.3
        self.init_inertia = [0.01, 0.012, 0.025]
        self.init_kp = [1.0, 1.2, 2.5]
        self.init_kd = [0.001, 0.001, 0.002]

        # GS model
        gs_dir = Path("assets/gs_data")
        resolution_quality = 0.4
        self.gs = get_gs(self.map_name, gs_dir, resolution_quality)

        self.init_sim()
        self.episode_length = episode_length

        # navigation parameters
        if self.map_name == 'gate_mid_new':
            target = (13.0, -2.0, 1.5) # in map absolute dimension, start with 0
            target_xy = (13.0, -2.0)
        else:
            raise ValueError(f"Unknown map name: {self.map_name}")
        
        self.target = tu.to_torch(list(target), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # navigation target
        self.target_xy = tu.to_torch(list(target_xy), 
            device=self.device, requires_grad=True).repeat((self.num_envs, 1)) # navigation target
        self.gs_origin_offset = torch.tensor([[-6.0, -0., -0.05]] * self.num_envs, device=self.device) # (x,y,z); gs dimension for visual info
        self.point_could_origin_offset = torch.tensor([[-6.0, -0., -0.05]] * self.num_envs, device=self.device) # (x,y,z); point cloud dimension for obstacle distance & traj plan

        # setup point cloud
        script_dir = Path(__file__).parent
        point_cloud_dir = script_dir / "assets" / "point_cloud"
        ply_file = point_cloud_dir / f"{maps[self.map_name]}.ply"
        self.point_cloud = ObstacleDistanceCalculator(ply_file=ply_file)
        self.traj_planner = TrajectoryPlanner(ply_file=ply_file, 
                                            safety_distance=0.15, 
                                            batch_size=1, 
                                            wp_distance=2.0,
                                            verbose=False)
        

        # generate ref_traj
        if self.map_name == 'gate_mid_new':
            traj_start = torch.tensor([[-6.0, 0.0, 1.4]], device=self.device)
            self.wp_num = 3
            self.ref_traj_size = 20
            self.task_table = {
                # Training tasks
                "[Task 0] Go straight through the gate then STOP over MONITOR": [torch.tensor([ # x, y, z
                                                                [2.0, 0.2, 1.4],
                                                                [5.8, 0.2, 1.4],
                                                                [6.5, -1.0, 1.6],
                                                                ], device=self.device)
                                                    ],
                "[Task 1] Go straight through the gate then FLY to LADDER base": [torch.tensor([ # x, y, z
                                                                [2.0, 0.2, 1.4],
                                                                [5.8, 0.2, 1.4],
                                                                [6.5, 1.2, 0.8],
                                                                ], device=self.device)
                                                    ],
                "[Task 2] FLY passed the RIGHT side of the gate then FLY to LADDER base": [torch.tensor([ # x, y, z
                                                                [2.5, -0.6, 1.35],
                                                                [5.0, -1.4, 1.4],
                                                                [8.0, 0.9, 1.1],
                                                                ], device=self.device)
                                                    ],
                "[Task 3] FLY passed the RIGHT side of the gate then STOP over CART": [torch.tensor([ # x, y, z
                                                                [2.5, -0.6, 1.4],
                                                                [5.0, -1.3, 1.4],
                                                                [10.5, 0.2, 1.7],
                                                                ], device=self.device)
                                                    ],
                "[Task 4] FLY passed the LEFT side of the gate then STOP over MONITOR": [torch.tensor([
                                                                [2.5, 0.8, 1.35],
                                                                [5.5, 1.8, 1.4],
                                                                [8.5, -0.6, 1.6],
                                                                ], device=self.device)
                                                    ],
                "[Task 5] FLY passed the LEFT side of the gate then STOP over CART": [torch.tensor([
                                                                [2.5, 0.8, 1.35],
                                                                [4.5, 1.8, 1.4],
                                                                [10.0, -0.5, 1.7],
                                                                ], device=self.device)
                                                    ],
                "[Task 6] FLY ABOVE the gate then FLY to LADDER base": [torch.tensor([
                                                                [4.0, 0.2, 2.2],
                                                                [5.9, 0.2, 2.5],
                                                                [8.0, 1.0, 1.5],
                                                                ], device=self.device)
                                                    ],
                "[Task 7] FLY ABOVE the gate then STOP over MONITOR": [torch.tensor([
                                                                [4.0, 0.2, 2.2],
                                                                [5.9, 0.2, 2.5],
                                                                [9.0, -1.0, 1.6],
                                                                ], device=self.device)
                                                    ],

                # # # Evaluation tasks
                # "[Task 8] GO THROUGH the gate. The gate must be passed. Go straight through the gate → STOP over CART": [torch.tensor([ # x, y, z
                #                                                 [2.0, 0.2, 1.4],
                #                                                 [5.8, 0.2, 1.4],
                #                                                 [10.5, 0.2, 1.7],
                #                                                 ], device=self.device)
                #                                     ],
                # "[Task 9] FLY passed the RIGHT side of the gate. Fly next to the right side of the gate. Pass the gate from the right → STOP over MONITOR": [torch.tensor([ # x, y, z
                #                                                 [2.5, -0.7, 1.4],
                #                                                 [5.0, -1.5, 1.4],
                #                                                 [8.5, -1.0, 1.6],
                #                                                 ], device=self.device)
                #                                     ],
                # "[Task 10] FLY passed the LEFT side of the gate. Fly next to the left side of the gate. Pass the gate from the left → FLY to LADDER base": [torch.tensor([
                #                                                 [2.5, 0.6, 1.35],
                #                                                 [4.5, 1.6, 1.4],
                #                                                 [8.0, 1.0, 0.8],
                #                                                 ], device=self.device)
                #                                     ],
                # "[Task 11] FLY ABOVE the gate. The gate is below. Fly directly over the gate → STOP over CART": [torch.tensor([
                #                                                 [4.0, 0.2, 2.1],
                #                                                 [5.9, 0.2, 2.4],
                #                                                 [10.5, 0.2, 1.7],
                #                                                 ], device=self.device)
                #                                     ],
            }
            
        else:
            raise ValueError(f"Unknown map name: {self.map_name}")
        
        # Generate reference trajectory
        for key, value in self.task_table.items():
            reward_wp = value[0]
            traj_wp = reward_wp + self.point_could_origin_offset[0].repeat((reward_wp.shape[0],1))
            traj_start = torch.tensor([[-6.0, 0, 1.4]], device=self.device)
            self.task_table[key].append(torch.tensor(self.traj_planner.plan_trajectories(traj_start, 
                                                                                            [traj_wp], 
                                                                                            des_wp_num=self.ref_traj_size)[0], 
                                                    device=self.device))

       
        # Multitask training simultaneously preparation
        task_names = list(self.task_table.keys())
        self.task_num = len(task_names)
        self.eval_task_id = eval_task_id
        if self.num_envs == 1:
            self.ini_task_indices = torch.tensor([self.eval_task_id], device=self.device)
        else:
            self.ini_task_indices = torch.randint(0, self.task_num, (self.num_envs,))
        
        self.task_list = [task_names[i] for i in self.ini_task_indices]
        self.reward_wp_tensor = torch.zeros((self.num_envs, self.wp_num, 3), device=self.device)
        self.ref_traj_tensor = torch.zeros((self.num_envs, self.ref_traj_size, 3), device=self.device)

        # Fill in waypoints based on task_list
        for i in range(self.num_envs):
            task_name = self.task_list[i]
            wp = self.task_table[task_name][0]  
            ref_traj = self.task_table[task_name][1] 
            self.reward_wp_tensor[i] = wp
            self.ref_traj_tensor[i] = ref_traj

        
        # TODO: implement a better way to handle the reward waypoints
        self.reward_wp_record = []
        for i in range(self.wp_num):
            self.reward_wp_record.append(torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        
        # Reward parameters
        if self.map_name == 'gate_mid_new':
            self.up_strength = 0.25
            self.heading_strength = 0.3 # reward yaw motion align with lin_vel
            self.lin_strength = -0.5
            self.lin_vel_rate_penalty = -0.75
            self.action_penalty = -1.0
            self.action_change_penalty = -1.0
            self.smooth_penalty = -0.75
            self.survive_reward = 8.0
            self.pose_penalty = -0.5
            self.height_penalty = -1.5
            self.height_change_penalty = -0.5
            self.jitter_penalty = -0.1
            self.out_map_penalty = -1.5
            self.map_center_penalty = -0.05
            # trajectory
            self.waypoint_strength = 4.0
            self.target_factor = -4.0
            # obstacle avoidance
            self.obstacle_strength = 1.0
            self.collision_penalty_coef = 0.0
            self.collision_penalty = self.collision_penalty_coef*self.survive_reward 
        else:
            raise ValueError(f"Unknown map name: {self.map_name}")
        
       
        self.obst_threshold = 0.5
        self.obst_collision_limit = 0.20
        self.body_rate_threshold = 15
        self.domain_randomization = False


        # self.dt = 0.05 # outter loop RL policy freaquency
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
        if self.use_depth:
            self.visual_net = VisualPerceptionNet(input_channels=1, visual_feature_size=self.visual_feature_size).to(self.device)
        else:
            self.visual_net = VisualPerceptionNet(input_channels=3, visual_feature_size=self.visual_feature_size).to(self.device)
        self.vlm = VLM(self.num_envs, self.device, vlm_feature_size=self.vlm_feature_dim).to(self.device)
        self.obs_hist_buf = ObsHistBuffer(batch_size=self.num_envs,
                                          vector_dim=self.num_latent_obs,
                                          buffer_size=self.num_history,
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
            "pose_penalty": self.pose_penalty,
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
            "resolution_quality": resolution_quality,
            "br_delay_factor": self.br_delay_factor,
            "thrust_delay_factor":self.thrust_delay_factor,
            "start_height": self.start_height,
            "target_height": self.target_height,
        }

        
        #-----------------------
        # set up Visualization
        self.time_stamp = get_time_stamp()
        self.episode_count = 0
        
        # recored x, y, depth for test
        self.x_record = [[] for _ in range(self.task_num)]
        self.y_record = [[] for _ in range(self.task_num)]
        self.z_record = [[] for _ in range(self.task_num)]
        self.roll_record = [[] for _ in range(self.task_num)]
        self.pitch_record = [[] for _ in range(self.task_num)]
        self.yaw_record = [[] for _ in range(self.task_num)]
        self.velo_x_record = [[] for _ in range(self.task_num)]
        self.velo_y_record = [[] for _ in range(self.task_num)]
        self.velo_z_record = [[] for _ in range(self.task_num)]
        self.ang_velo_x_record = [[] for _ in range(self.task_num)]
        self.ang_velo_y_record = [[] for _ in range(self.task_num)]
        self.ang_velo_z_record = [[] for _ in range(self.task_num)]
        self.vae_velo_x_record = [[] for _ in range(self.task_num)]
        self.vae_velo_y_record = [[] for _ in range(self.task_num)]
        self.vae_velo_z_record = [[] for _ in range(self.task_num)]
        self.depth_record = [[] for _ in range(self.task_num)]
        self.img_record = [[] for _ in range(self.task_num)]
        self.action_record = np.zeros([self.task_num,self.episode_length,4])
        self.done_record = np.zeros([self.task_num])
        self.reward_record = torch.zeros([self.task_num], device=self.device)


        
        
    def create_saving_folder(self):
        if (self.visualize):
            curr_path = os.getcwd()
            if self.num_envs == 1:
                self.save_indices = [0]
                self.save_path = [f'{curr_path}/examples/outputs/{self.agent_name}/{self.actor_nn_name}/{self.map_name}/NLP/task_{self.eval_task_id}/{self.time_stamp}_individual_eval']
            else:
                self.select_eval_indices()
                self.save_path = []
                for i in range(self.task_num):
                    self.save_path.append(f'{curr_path}/examples/outputs/{self.agent_name}/{self.actor_nn_name}/{self.map_name}/NLP/task_{i}/{self.time_stamp}')
            
            # create saving folder
            for i in range(len(self.save_path)):
                os.makedirs(self.save_path[i], exist_ok=True)

    def select_eval_indices(self):
        self.save_indices = [-1]*self.task_num
        i = 0
        while -1 in self.save_indices:
            id = self.ini_task_indices[i]
            if self.save_indices[id] == -1:
                self.save_indices[id] = i
            i += 1
            if i >= self.num_envs:
                raise ValueError("Not enough indices to assign to all tasks.")
            



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

        if self.map_name == 'gate_mid_new':
            self.start_rot = np.array([0., 0., 0., 1.]) # (x,y,z,w)
        else:
            raise ValueError(f"Unknown map name: {self.map_name}")
        self.start_rotation = tu.to_torch(self.start_rot, device=self.device, requires_grad=False)

        # initialize some data used later on
        self.up_vec = self.z_unit_tensor.clone()
        self.heading_vec = self.x_unit_tensor.clone()
        self.inv_start_rot = tu.quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        # Initialize the drone dynamics parameters
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
            br_noise_std=0.01,  # Scalar
        )
        
        # set start pos
        if self.map_name == 'gate_mid_new':
            start_pos_x = 0.0; start_pos_y = 0.0
            self.start_body_rate = [0., 0., 0.]
        else:
            raise ValueError(f"Unknown map name: {self.map_name}")

        self.start_pos = []
        self.start_norm_thrust = [(self.hover_thrust / self.max_thrust).clone().detach().cpu().numpy()[0]]
        self.control_base = self.start_norm_thrust[0]
        self.start_action = self.start_body_rate + self.start_norm_thrust
        

        for i in range(self.num_environments):
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
        
        self.latent_vect = torch.zeros([self.num_envs, self.num_latent], device=self.device, dtype = torch.float)
        self.lin_vel_vae = torch.zeros([self.num_envs, 3], device=self.device, dtype = torch.float)
        self.prev_lin_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype = torch.float)

        # VLM variables
        self.vlm_feature = torch.zeros([self.num_envs, self.vlm_feature_dim], device=self.device, dtype = torch.float)
        
    def set_task_eval(self, task_id):
        """
        Set the task for evaluation. Only work when evaluation num == 1. 
        Otherwise, save all the tasks in the task_list and use the task_list to sample
        Args:
            task_id: int, the task id to be evaluated
        """
        # self.eval_task_id = task_id
        if self.num_envs == 1:
            self.ini_task_indices = torch.tensor([task_id], device=self.device)
            self.task_list = [list(self.task_table.keys())[task_id]] * self.num_envs
            self.reward_wp_tensor = torch.zeros((self.num_envs, self.wp_num, 3), device=self.device)
            self.ref_traj_tensor = torch.zeros((self.num_envs, self.ref_traj_size, 3), device=self.device)

            # Fill in waypoints based on task_list
            for i in range(self.num_envs):
                task_name = self.task_list[i]
                wp = self.task_table[task_name][0]  
                ref_traj = self.task_table[task_name][1] 
                self.reward_wp_tensor[i] = wp
                self.ref_traj_tensor[i] = ref_traj
    
    # Forward simulation of the trajectory
    def step(self, actions, vae_info):
        self.latent_vect = vae_info.clone().detach()

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
            self.vlm_feature_before_reset = self.vlm_feature.clone()
            self.extras = {
                'obs_before_reset': self.obs_buf_before_reset,
                'privilege_obs_before_reset': self.privilege_obs_buf_before_reset,
                'episode_end': self.termination_buf,
                "vlm_feature_before_reset": self.vlm_feature_before_reset,
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

        return self.obs_buf, self.privilege_obs_buf, obs_hist, obs_vel, self.rew_buf, self.reset_buf, self.vlm_feature, self.extras
    
    

    def reset(self, env_ids = None, force_reset = True):
        if env_ids is None:
            if force_reset == True:
                env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        if env_ids is not None:
            num_reset_envs = len(env_ids)
            if not self.visualize and self.domain_randomization:
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
                    batch_size=self.num_envs,  # Number of drones
                    mass=self.mass,  # Shape: (batch_size,)
                    inertia=self.inertia,  # Shape: (batch_size, 3, 3)
                    link_length=0.15,  # Scalar
                    Kp=torch.tensor(self.init_kp, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),  # Shape: (batch_size, 3)
                    Kd=torch.tensor(self.init_kd, device=self.device).unsqueeze(0).repeat(self.num_envs, 1),  # Shape: (batch_size, 3)
                    freq=200.0,  # Scalar
                    max_thrust=self.max_thrust,  # Shape: (batch_size,)
                    total_time=self.sim_dt,  # Scalar
                    rotor_noise_std=0.01,  # Scalar
                    br_noise_std=0.01,  # Scalar
                    motor_time_constant=0.1,
                    torque_constant=0.05,
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
            self.latent_vect[env_ids] = torch.zeros([len(env_ids), self.num_latent], device=self.device, dtype = torch.float)
            self.lin_vel_vae[env_ids] = torch.zeros([len(env_ids), 3], device=self.device, dtype = torch.float)

            # Reset VLM variables
            if not self.visualize:
                task_names = list(self.task_table.keys())
                task_indices = torch.randint(0, self.task_num, (num_reset_envs,))
                
                # Fill in waypoints based on task_list
                for i in range(num_reset_envs):
                    task_name = task_names[task_indices[i].item()]
                    self.task_list[env_ids[i]] = task_name
                    wp = self.task_table[task_name][0]  
                    ref_traj = self.task_table[task_name][1] 
                    self.reward_wp_tensor[env_ids[i]] = wp
                    self.ref_traj_tensor[env_ids[i]] = ref_traj

            
            self.progress_buf[env_ids] = 0
            self.calculateObservations()

        return self.obs_buf, self.vlm_feature
    
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

        return self.obs_buf, self.privilege_obs_buf, self.vlm_feature

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
    
    def find_factors(self, n):
        """Finds two approximate factors (h, w) of n that are close to a square shape."""
        h = int(math.sqrt(n))
        while n % h != 0:
            h -= 1
        w = n // h
        return h, w

    # Parralized implementation
    def process_GS_data(self, depth_list, nerf_img):
        # min depth from nerf
        batch,H,W,ch = depth_list.shape
        depth_list_up = depth_list[:,0:int(H/2),:,:]
        self.depth_list = torch.abs(torch.amin(depth_list_up,dim=(1,2,3))) 
        self.depth_list = self.depth_list.unsqueeze(1)
        self.depth_list = self.depth_list.to(device=self.device)

        rgb_tensor = nerf_img.permute(0, 3, 1, 2)
        depth_tensor = depth_list.permute(0, 3, 1, 2)  # Make sure depth_list is [batch_size, channels, height, width]
        depth_tensor = torch.clip(depth_tensor, 0.0, 2.5) # match the realsense hardware limit

        resize = nn.AdaptiveAvgPool2d((224, 224))
        depth_tensor = resize(depth_tensor).to(self.device)
        rgb_tensor = resize(rgb_tensor).to(self.device)

        # Visual network feature extraction
        if self.use_depth:
            # Only use depth
            h, w = self.find_factors(self.visual_feature_size)
            pooled = -F.max_pool2d(depth_tensor, (h, w)) # minimum pooling depth for collision avoidance
            pooled = F.adaptive_avg_pool2d(pooled, (1, self.visual_feature_size))
            self.visual_info = pooled.view(batch, self.visual_feature_size)
        else:
            # Only use RGB
            self.visual_info = self.visual_net(rgb_tensor).detach()

        # VLM feature
        # Compute mask for environments where progress % 10 == 0
        mask = (self.progress_buf % 10 == 0)  # shape: [batch_size], dtype: torch.bool
        # Only update VLM feature for masked environments
        if mask.any():
            indices = mask.nonzero(as_tuple=True)[0]
            selected_prompts = [self.task_list[i] for i in indices.tolist()] 
            new_vlm_features = self.vlm(rgb_tensor[mask], selected_prompts).detach()
            if not hasattr(self, 'vlm_feature') or self.vlm_feature is None:
                self.vlm_feature = torch.zeros((batch, new_vlm_features.shape[1]), device=self.device)
            # Update only masked environments
            self.vlm_feature[mask] = new_vlm_features




    def add_noise_to_vlm_feature(self, vlm_feature, std=0.01, training=True, mode='additive'):
        """
        Adds small noise to VLM feature for regularization.
        
        Args:
            vlm_feature (torch.Tensor): [B, D] VLM embedding (from projection).
            std (float): standard deviation of noise.
            training (bool): whether the model is in training mode.
            mode (str): 'additive' for simple Gaussian noise,
                        'directional' for unit-norm sphere perturbation.
        
        Returns:
            torch.Tensor: noisy VLM feature (same shape).
        """
        if not training or std <= 0.0:
            return vlm_feature

        if mode == 'additive':
            noise = torch.randn_like(vlm_feature) * std
            return vlm_feature + noise

        elif mode == 'directional':
            unit_noise = F.normalize(torch.randn_like(vlm_feature), dim=-1)
            return F.normalize(vlm_feature + std * unit_noise, dim=-1)

        else:
            raise ValueError(f"Unknown noise mode: {mode}")
        


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
                self.time_report.end_timer("3D GS inference")
                self.process_GS_data(depth_list, nerf_img)
            self.nerf_count += 1            
            
        # Draw record plot for data visualization
        if self.visualize:
            img_transform = Resize((360, 640), antialias=True)

            # ---------------------------------------
            # Normal test mode
            # 3DGS data
            depth_list, nerf_img = self.gs.render(gs_pose) # (batch_size,H,W,1/3)
            self.process_GS_data(depth_list, nerf_img)

            img_batch = []
            for i, id in enumerate(self.save_indices):
                nerf_img_i = nerf_img[id].permute(2, 0, 1).cpu()  # [C, H, W] on CPU
                resized = img_transform(nerf_img_i)              # [C, 360, 640]
                img_uint8 = (resized * 255).clamp(0, 255).to(torch.uint8)
                self.img_record[i].append(img_uint8)             # Save as uint8 to save RAM

            pos_data = torso_pos.clone()
            pos_data = pos_data.detach().cpu().numpy()
            depth_data = self.depth_list.clone()
            depth_data = depth_data.detach().cpu().numpy()
            action_data = self.actions.clone().detach().cpu().numpy()
            ang_vel_data = ang_vel.clone().detach().cpu().numpy()
            vae_velo_data = self.lin_vel_vae.clone().detach().cpu().numpy()
            velo_data = lin_vel.clone().detach().cpu().numpy()

            for i, env_id in enumerate(self.save_indices):
                self.x_record[i].append(pos_data[env_id, 0])
                self.y_record[i].append(pos_data[env_id, 1])
                self.z_record[i].append(pos_data[env_id, 2])
                self.roll_record[i].append(rpy_data[env_id, 0])
                self.pitch_record[i].append(rpy_data[env_id, 1])
                self.yaw_record[i].append(rpy_data[env_id, 2])
                self.velo_x_record[i].append(velo_data[env_id, 0])
                self.velo_y_record[i].append(velo_data[env_id, 1])
                self.velo_z_record[i].append(velo_data[env_id, 2])
                self.ang_velo_x_record[i].append(ang_vel_data[env_id, 0])
                self.ang_velo_y_record[i].append(ang_vel_data[env_id, 1])
                self.ang_velo_z_record[i].append(ang_vel_data[env_id, 2])
                self.vae_velo_x_record[i].append(vae_velo_data[env_id, 0])
                self.vae_velo_y_record[i].append(vae_velo_data[env_id, 1])
                self.vae_velo_z_record[i].append(vae_velo_data[env_id, 2])
                self.depth_record[i].append(depth_data[env_id] / 2)
                if self.episode_count < self.episode_length:
                    self.action_record[i, self.episode_count, :] = action_data[env_id]

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
        vlm_noise = self.obs_noise_level * (torch.rand_like(self.vlm_feature) - 0.5)
        latent_noise = self.obs_noise_level * (torch.rand_like(self.latent_vect) - 0.5)
        torso_quat_noise = self.obs_noise_level * (torch.rand_like(torso_quat) - 0.5)

        self.privilege_obs_buf = torch.cat([
                                torso_pos[:, :], # 0:3 
                                lin_vel, # 3:6
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
        # pose_penalty = torch.sum(torch.abs(self.obs_buf[:, 2:6] - self.start_rotation), dim=-1) * self.pose_penalty
        pose_penalty = torch.sum(torch.square(self.obs_buf[:, 2:6] - self.start_rotation), dim=-1) * self.pose_penalty
        height_penalty = torch.square(self.obs_buf[:,0] - self.target_height) * self.height_penalty
        height_change_penalty = torch.square(self.obs_buf[:,1]) * self.height_change_penalty
        jitter_penalty = torch.square(self.privilege_obs_buf[:,8]) * self.jitter_penalty 
        out_map_penalty = (
            torch.clip(self.map_x_min - self.privilege_obs_buf[:, 0], min=0) ** 2 +
            torch.clip(self.privilege_obs_buf[:, 0] - self.map_x_max, min=0) ** 2 +
            torch.clip(self.map_y_min - self.privilege_obs_buf[:, 1], min=0) ** 2 +
            torch.clip(self.privilege_obs_buf[:, 1] - self.map_y_max, min=0) ** 2
        ) * self.out_map_penalty 
        map_center_penalty = torch.square(self.privilege_obs_buf[:,2]) * self.map_center_penalty 

        # Navigation related rewards
        wp_positions = self.privilege_obs_buf[:, 0:3].unsqueeze(1)  # [num_envs, 1, 3]
        waypoints = self.reward_wp_tensor  # [num_envs, 3, 3]
        distances = torch.norm(wp_positions - waypoints, dim=2)**2  # [num_envs, 3]
        factors = torch.exp(1 / (distances + 1.0)) - 1  # [num_envs, 3]
        wp_reward = (factors.sum(dim=1)) * self.waypoint_strength  # [num_envs]

        # tracking reference trajectory
        ref_x = self.privilege_obs_buf[:, 0] + self.point_could_origin_offset[0, 0]  # [num_envs]
        ref_traj_x = self.ref_traj_tensor[:, :, 0]  # [num_envs, 10]
        diff = ref_traj_x - ref_x.unsqueeze(1)  # [num_envs, 10]
        mask = diff > 0.5  # [num_envs, 10]
        diff_masked = torch.where(mask, diff, float('inf'))  # [num_envs, 10]
        min_diffs, min_indices = torch.min(diff_masked, dim=1)  # [num_envs], [num_envs]
        no_valid_indices = torch.isinf(min_diffs)  # [num_envs]
        target_list = torch.empty((self.num_envs, 3), device=self.device)
        default_target = self.target[0] + self.point_could_origin_offset[0]  # [3]
        target_list[:] = default_target
        valid_indices = ~no_valid_indices
        selected_indices = min_indices[valid_indices]  # [num_valid_envs]
        env_indices = torch.arange(self.num_envs, device=self.device)[valid_indices]  # [num_valid_envs]
        targets = self.ref_traj_tensor[env_indices, selected_indices]  # [num_valid_envs, 3]
        target_list[valid_indices] = targets
        target_list = target_list - self.point_could_origin_offset  # [num_envs, 3]
        curr_pos = self.privilege_obs_buf[:, 0:3]  # [num_envs, 3]
        desire_velo_norm = (target_list - curr_pos) / torch.clip(torch.norm(target_list - curr_pos, dim=1, keepdim=True), min=1e-6)  # [num_envs, 3]
        curr_velo = self.obs_buf[:, 14:17]  # [num_envs, 3]
        curr_velo_norm = curr_velo / torch.clip(torch.norm(curr_velo, dim=1, keepdim=True), min=1e-2)  # [num_envs, 3]
        velo_dist = torch.norm(curr_velo_norm - desire_velo_norm, dim=1)  # [num_envs]
        target_reward = velo_dist * self.target_factor  # [num_envs]

        # Obstacle avoidance reward
        obst_reward = torch.tensor([0.], requires_grad=True, device=self.device)
        self.time_report.start_timer("point cloud collision check")
        obst_dist = self.point_cloud.compute_nearest_distances(drone_pos, drone_rot)
        self.time_report.end_timer("point cloud collision check")
        # reward larger distance away from obstacles
        obst_reward = obst_reward + torch.where(obst_dist < self.obst_threshold, obst_dist*self.obstacle_strength, torch.zeros_like(obst_dist))
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
            for i, env_id in enumerate(self.save_indices):
                if self.reset_buf[env_id] == 1:
                    self.done_record[i] = 1
                    self.reward_record[i] -= self.rew_buf[env_id]
                if self.done_record[i] == 0:
                    self.reward_record[i] -= self.rew_buf[env_id]

            if self.episode_count == (self.episode_length-1) or combined_condition.any():
                self.save_recordings_batch()

            



    def save_recordings_batch(self):
        for i, save_path in enumerate(self.save_path):
            os.makedirs(save_path, exist_ok=True)
            map_id = self.save_indices[i]

            # Save the task instruction and final reward in txt
            task_name = self.task_list[map_id]
            final_reward = self.reward_record[i]
            with open(f'{save_path}/task_overview.txt', 'w', encoding='utf-8') as f:
                f.write(f'Task: {task_name}\n')
                f.write(f'Final loss: {final_reward}\n')

            # Save raw data
            np.save(f'{save_path}/x.npy', np.array(self.x_record[i], dtype=object), allow_pickle=True)
            np.save(f'{save_path}/y.npy', np.array(self.y_record[i], dtype=object), allow_pickle=True)
            np.save(f'{save_path}/z.npy', np.array(self.z_record[i], dtype=object), allow_pickle=True)

            # Z plot
            plt.figure()
            plt.plot(range(len(self.z_record[i])), self.z_record[i])
            plt.xlabel("Step")
            plt.ylabel("Z/m")
            plt.savefig(f'{save_path}/z_plot.png')
            plt.close()

            # XYZ position plot
            plt.figure()
            plt.plot(range(len(self.x_record[i])), self.x_record[i])
            plt.plot(range(len(self.y_record[i])), self.y_record[i])
            plt.plot(range(len(self.z_record[i])), self.z_record[i])
            plt.xlabel("Step")
            plt.ylabel("m")
            plt.legend(['x', 'y', 'z'])
            plt.savefig(f'{save_path}/pos_plot.png')
            plt.close()

            # Orientation plot
            plt.figure()
            plt.plot(range(len(self.roll_record[i])), self.roll_record[i])
            plt.plot(range(len(self.pitch_record[i])), self.pitch_record[i])
            plt.plot(range(len(self.yaw_record[i])), self.yaw_record[i])
            plt.xlabel("Step")
            plt.ylabel("rad")
            plt.legend(['r', 'p', 'y'])
            plt.savefig(f'{save_path}/pose_plot.png')
            plt.close()

            # Velocity plot
            plt.figure()
            plt.plot(range(len(self.velo_x_record[i])), self.velo_x_record[i])
            plt.plot(range(len(self.velo_y_record[i])), self.velo_y_record[i])
            plt.plot(range(len(self.velo_z_record[i])), self.velo_z_record[i])
            plt.plot(range(len(self.vae_velo_x_record[i])), self.vae_velo_x_record[i], linestyle='dashed')
            plt.plot(range(len(self.vae_velo_y_record[i])), self.vae_velo_y_record[i], linestyle='dashed')
            plt.plot(range(len(self.vae_velo_z_record[i])), self.vae_velo_z_record[i], linestyle='dashed')
            plt.xlabel("Step")
            plt.ylabel("m/s")
            plt.legend(['vx', 'vy', 'vz', 'vae_vx', 'vae_vy', 'vae_vz'])
            plt.savefig(f'{save_path}/velo_plot.png')
            plt.close()

            # Depth plot
            plt.figure()
            plt.plot(range(len(self.depth_record[i])), self.depth_record[i])
            plt.xlabel("Step")
            plt.ylabel("depth/m")
            plt.savefig(f'{save_path}/depth_plot.png')
            plt.close()

            # Action plot
            plt.figure()
            for action_id in range(3):
                plt.plot(range(np.shape(self.action_record[i])[0]), self.action_record[i][:, action_id], label=f'rotor {action_id}')
            plt.plot(range(len(self.ang_velo_x_record[i])), self.ang_velo_x_record[i], linestyle='dashed')
            plt.plot(range(len(self.ang_velo_y_record[i])), self.ang_velo_y_record[i], linestyle='dashed')
            plt.plot(range(len(self.ang_velo_z_record[i])), self.ang_velo_z_record[i], linestyle='dashed')
            plt.xlabel('Step')
            plt.ylabel('body_rate')
            plt.legend(['r_des', 'p_des', 'y_des', 'r', 'p', 'y'])
            plt.savefig(f'{save_path}/body_rate_plot.png')
            plt.close()

            # Save as .txt
            np.savetxt(f'{save_path}/action.txt', self.action_record[i], delimiter=',')

            # Thrust force plot
            plt.figure()
            plt.plot(range(np.shape(self.action_record[i])[0]), self.action_record[i][:, 3])
            plt.xlabel('Step')
            plt.ylabel('normalized_force')
            plt.savefig(f'{save_path}/action_force_plot.png')
            plt.close()

            # Save video
            video_tensor = torch.permute(torch.stack(self.img_record[i]), (0, 2, 3, 1)) 
            video_tensor = video_tensor.to('cpu').to(dtype=torch.uint8)
            write_video(f'{save_path}/ego_video.mp4', video_tensor, fps=20)

            # Save first image
            ini_img = torch.permute(self.img_record[i][0], (1, 2, 0)) 
            ini_img = ini_img.clone().detach().cpu().numpy().astype(np.uint8)
            ini_img = Image.fromarray(ini_img)
            ini_img.save(f'{save_path}/ini_img.png')

            

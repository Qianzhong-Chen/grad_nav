from math import e
from pathlib import Path
import os
from re import T
import torch
import numpy as np
import yaml
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from scipy.spatial.transform import Rotation as R
from nerfstudio.utils.poses import multiply
from typing import Literal, List
import matplotlib.pyplot as plt
import time
# import synthesize.trajectory_helper as th
torch.set_float32_matmul_precision('high')

class GS():
    def __init__(self, config_path: Path, width:int=640, height:int=360, res:float = 0.15) -> None:
        # config path
        self.config_path = config_path
        print(self.config_path)

        # Pytorch Config
        use_cuda = torch.cuda.is_available()                    
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        # Get config and pipeline
        self.config, self.pipeline, _, _ = eval_setup(
            self.config_path, 
            test_mode="inference",
        )

        # Get reference camera
        self.camera_ref = self.pipeline.datamanager.eval_dataset.cameras[0]

        # Render parameters
        self.channels = 3
        self.camera_out,self.width,self.height = self.generate_output_camera(width,height)
        self.camera_out.rescale_output_resolution(res)
       
    def generate_output_camera(self,width:int,height:int):
        fx,fy = 462.956,463.002
        cx,cy = 323.076,181.184
        
        camera_out = Cameras(
            camera_to_worlds=1.0*self.camera_ref.camera_to_worlds,
            fx=fx,fy=fy,
            cx=cx,cy=cy,
            width=width,
            height=height,
            camera_type=CameraType.PERSPECTIVE,
        )
        camera_out = camera_out.to(self.device)

        return camera_out,width,height
    
        
    def render_quad_poses(self, ros_poses: torch.Tensor) -> List[torch.Tensor]:
        """
        Render from a batch of quad poses

        Args:
            ros_poses (torch.Tensor): expected shape (N, 3, 4)
        Returns:

        """
        ns_poses = self.convert_poses_quad2ns(ros_poses)
        num_envs = ros_poses.size(0)
        depth_list = []
        img_list = []
        for i in range(ns_poses.shape[0]):
            pose = ns_poses[i].unsqueeze(0)
            # depth[i, :], img =self.render(pose)
            depth, img =self.render(pose)
            depth_list.append(depth)
            img_list.append(img)
        return torch.stack(depth_list,dim=0), torch.stack(img_list,dim=0)
    
    def convert_poses_quad2ns(self, ros_poses: torch.Tensor) -> torch.Tensor:
        """
        Convert from a quad pose convention to a Nerfstudio pose.
            1. Convert coordinate systems from ROS to Nerfstudio
            2. Transform with the Nerfstudio dataparser transform and scale

        Args:
            poses (torch.Tensor): expected shape (N, 3, 4)

        Returns:
            torch.Tensor: expected shape (N, 3, 4)
        """
        T_ns = ros_poses[:, :, [1, 2, 0, 3]]
        T_ns[:, :, [0, 2]] *= -1
        T_ns = multiply(self.dataparser_T, T_ns)
        T_ns[:, :, 3] *= self.dataparser_scale
        return T_ns

    def render(self, xcr_list: torch.Tensor) -> tuple:
        """
        Render from multiple poses in xcr_list and return a batched output.

        Args:
            xcr_list (torch.Tensor): A tensor of shape (n, 10) where n is the batch size.

        Returns:
            tuple: A tuple (depth_batch, img_batch) where each is a tensor of shape (n, H, W, channels).
        """
        # Get batch size and set device
        batch_size = xcr_list.size(0)
        device = xcr_list.device
        
        # Obtain position and quaternion batches
        positions = xcr_list[:, :3]  # Shape: (n, 3)
        quaternions = xcr_list[:, 6:10]  # Shape: (n, 4)
        
        # Compute batched transformation matrices in parallel
        T_c2n_batch = pose2nerf_transform(positions, quaternions, device)  # Shape: (n, 4, 4)
        P_c2n_batch = T_c2n_batch[:, :3, :].float()  # Shape: (n, 3, 4)

        # Pre-allocate tensors for batched depth and img outputs
        depth_batch = []
        img_batch = []

        for i in range(batch_size):
            self.camera_out.camera_to_worlds = P_c2n_batch[i][None, :3, ...]
            # Render outputs
            outputs = self.pipeline.model.get_outputs_for_camera(self.camera_out, obb_box=None)
            img_batch.append(outputs["rgb"])
            depth_batch.append(outputs["depth"])

        return torch.stack(depth_batch,dim=0), torch.stack(img_batch,dim=0)


def get_gs(map: str, gs_path, resolution) -> 'GS':
    workspace_path = os.path.dirname(os.path.dirname(__file__))
    gs_data_root = os.path.join(workspace_path, "envs/assets/gs_data")
    maps = {
        "gate_left": "sv_917_3_left_nerfstudio",
        "gate_right": "sv_917_3_right_nerfstudio",
        "gate_mid": "sv_1007_gate_mid",
    }

    map_folder = os.path.join(gs_data_root, maps[map])
    for root, _, files in os.walk(map_folder):
        if 'config.yml' in files:
            nerf_cfg_path = os.path.join(root, 'config.yml')
            break
    else:
        raise FileNotFoundError("config.yml not found in {}".format(map_folder))
    with open(nerf_cfg_path, "r") as f:
        content = f.read()
    content = content.replace("${GS_DATA_DIR}", gs_data_root)
    if "${" in content:
        raise ValueError("Unresolved environment variable found in config: {}".format(nerf_cfg_path))
    tmp_cfg_path = os.path.join(workspace_path, "temp_config_resolved.yaml")
    with open(tmp_cfg_path, "w") as f:
        f.write(content)
    main_dir_path = os.getcwd()
    os.chdir(gs_data_root)
    gs = GS(Path(tmp_cfg_path), res=resolution)
    os.chdir(main_dir_path)
    os.remove(tmp_cfg_path)

    return gs


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternions to rotation matrices.

    Args:
        quaternions (torch.Tensor): Tensor of shape (n, 4), where each quaternion is in (w, x, y, z) format.

    Returns:
        torch.Tensor: Rotation matrices of shape (n, 3, 3).
    """
    # Normalize the quaternions to ensure valid rotation
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Calculate the rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation_matrices = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=-1).view(-1, 3, 3)

    return rotation_matrices

def pose2nerf_transform(positions: torch.Tensor, quaternions: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Computes batched transformations from positions and quaternions.

    Args:
        positions (torch.Tensor): Tensor of shape (n, 3) for position vectors.
        quaternions (torch.Tensor): Tensor of shape (n, 4) for quaternion rotations.
        device (torch.device): Device for computation.

    Returns:
        torch.Tensor: A batched transformation matrix of shape (n, 4, 4).
    """
    batch_size = positions.size(0)

    # Fixed transformation matrices
    T_r2d = torch.tensor([
        [0.990, 0.000, 0.140, 0.152],
        [0.000, 1.000, 0.000, -0.031],
        [-0.140, 0.000, 0.990, -0.012],
        [0.000, 0.000, 0.000, 1.000]
    ], dtype=torch.float32, device=device).expand(batch_size, -1, -1)  # Shape: (n, 4, 4)

    T_f2n = torch.tensor([
        [1.000, 0.000, 0.000, 0.000],
        [0.000, -1.000, 0.000, 0.000],
        [0.000, 0.000, -1.000, 0.000],
        [0.000, 0.000, 0.000, 1.000]
    ], dtype=torch.float32, device=device).expand(batch_size, -1, -1)  # Shape: (n, 4, 4)

    T_c2r = torch.tensor([
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device).expand(batch_size, -1, -1)  # Shape: (n, 4, 4)

    
    rotation_matrices = quaternion_to_rotation_matrix(quaternions)
    T_d2f = torch.eye(4, device=device).expand(batch_size, -1, -1).clone()  # Initialize T_d2f
    T_d2f[:, :3, :3] = rotation_matrices  # Apply rotation
    T_d2f[:, :3, 3] = positions  # Apply translation

    # Combine all transformations
    T_c2n_batch = T_f2n @ T_d2f @ T_r2d @ T_c2r  # Shape: (n, 4, 4)

    return T_c2n_batch



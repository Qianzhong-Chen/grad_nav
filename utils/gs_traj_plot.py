from math import e
from pathlib import Path
import os
from re import T
import torch
import numpy as np
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from scipy.spatial.transform import Rotation as R
from nerfstudio.utils.poses import multiply
from typing import Literal, List
import matplotlib.pyplot as plt
import time
# import synthesize.trajectory_helper as th
torch.set_float32_matmul_precision('high')
import pdb

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
        # self.dataparser_T = (
        #     self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        # )
        # self.dataparser_scale = (
        #     self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale
        # )
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
        # start_t = time.time()
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
            
            
            # Set camera transformation for each batch
            self.camera_out.camera_to_worlds = P_c2n_batch[i][None, :3, ...]
            
            # Render outputs
            outputs = self.pipeline.model.get_outputs_for_camera(self.camera_out, obb_box=None)
            img_batch.append(outputs["rgb"])
            depth_batch.append(outputs["depth"])

        # Log render time for whole batch
        # elapsed_t = time.time() - start_t
        # print(f'{elapsed_t} s')
        return torch.stack(depth_batch,dim=0), torch.stack(img_batch,dim=0)


    def set_trajectory(self, trajectory_points: torch.Tensor):
        """Store trajectory for rendering."""
        self.trajectory_points = trajectory_points.to(self.device)

    def render_with_trajectory(self, xcr_list: torch.Tensor, trajectory_path: str, waypoints: torch.Tensor) -> tuple:
        """
        Render scene with a tube-like trajectory and waypoints overlaid from a saved file.

        Args:
            xcr_list (torch.Tensor): List of camera poses (N, 10).
            trajectory_path (str): Path to folder containing saved trajectory.
            waypoints (torch.Tensor): List of waypoint positions.

        Returns:
            tuple: (depth_batch, img_batch_with_overlay)
        """
        depth_batch, img_batch = self.render(xcr_list)

        # Load trajectory from file
        trajectory = self.load_trajectory(trajectory_path)

        # Overlay trajectory as a tube and waypoints as large markers
        img_batch = self.overlay_tube_trajectory_with_waypoints(img_batch, trajectory, waypoints)

        return depth_batch, img_batch


    def overlay_tube_trajectory_with_waypoints(self, image_batch, trajectory, waypoints):
        """
        Overlay a tube-like trajectory and larger waypoints on the rendered image using OpenCV.
        Fixes coordinate frame issues.

        Args:
            image_batch (torch.Tensor): Rendered image batch (N, H, W, 3).
            trajectory (torch.Tensor): Trajectory points (M, 3).
            waypoints (torch.Tensor): Waypoints (K, 3).

        Returns:
            torch.Tensor: Image with overlaid trajectory and waypoints.
        """
        import cv2

        image_np = (image_batch.cpu().numpy() * 255).astype(np.uint8)
        batch_size, h, w, _ = image_np.shape

        # Apply correct transformation to NeRF camera frame
        # trajectory = self.transform_to_nerf_camera_frame(trajectory)
        # waypoints = self.transform_to_nerf_camera_frame(waypoints)

        # Use correct intrinsics from NeRF config
        fx, fy = 900.5858, 896.4459
        cx, cy = 636.2945, 363.4638

        def project_points(points):
            """ Projects 3D world coordinates into 2D image plane using correct intrinsics. """
            projected_points = []
            for point in points.cpu().numpy():
                x, y, z = point
                u = int(fx * x / (z + 1e-6) + cx)
                v = int(fy * y / (z + 1e-6) + cy)
                if 0 <= u < w and 0 <= v < h:
                    projected_points.append((u, v))
            return projected_points

        # Project trajectory and waypoints
        projected_trajectory = project_points(trajectory)
        projected_waypoints = project_points(waypoints)

        # Draw trajectory as a thin green tube
        for i in range(batch_size):
            for j in range(len(projected_trajectory) - 1):
                cv2.line(image_np[i], projected_trajectory[j], projected_trajectory[j + 1], (0, 255, 0), 2)
                cv2.circle(image_np[i], projected_trajectory[j], radius=4, color=(0, 255, 0), thickness=-1)

            # Draw waypoints as large red circles
            for pt in projected_waypoints:
                cv2.circle(image_np[i], pt, radius=12, color=(0, 0, 255), thickness=-1)

        return torch.tensor(image_np, dtype=torch.float32) / 255.0

    def load_trajectory(self, trajectory_path: str) -> torch.Tensor:
        """
        Load trajectory from saved .npy files.
        
        Args:
            trajectory_path (str): Path to the folder containing x.npy, y.npy, z.npy.

        Returns:
            torch.Tensor: Loaded trajectory as (N, 3) tensor.
        """
        traj_x = np.load(os.path.join(trajectory_path, 'x.npy'), allow_pickle=True) - 6.0
        traj_y = np.load(os.path.join(trajectory_path, 'y.npy'), allow_pickle=True)
        traj_z = np.load(os.path.join(trajectory_path, 'z.npy'), allow_pickle=True) - 0.2

        traj_x = np.array(traj_x, dtype=np.float32)
        traj_y = np.array(traj_y, dtype=np.float32)
        traj_z = np.array(traj_z, dtype=np.float32)

        trajectory = np.stack([traj_x, traj_y, -traj_z], axis=1)
        # pdb.set_trace()
        return torch.tensor(trajectory, dtype=torch.float32, device=self.device)
    
    def transform_to_nerf_camera_frame(self, trajectory):
        """
        Transform the 3D trajectory from world frame to NeRF's camera frame.
        
        Args:
            trajectory (torch.Tensor): Trajectory points (N, 3) in world coordinates.

        Returns:
            torch.Tensor: Transformed trajectory points (N, 3) in NeRF's camera frame.
        """
        # Transformation from Open3D world frame to NeRF camera frame
        # NeRF camera frame: X-right, Y-down, Z-forward
        T_nerf = torch.tensor([
            [1,  0,  0],  # X -> X (unchanged)
            [0,  0, -1],  # Y -> -Z
            [0,  1,  0]   # Z -> Y
        ], dtype=torch.float32, device=self.device)

        return torch.matmul(trajectory, T_nerf.T)




def get_gs(map:str, gs_path, resolution) -> GS:
    # Generate some useful paths
    workspace_path = os.path.dirname(os.path.dirname(__file__))
    main_dir_path = os.getcwd()
    gs_path = os.path.join(workspace_path,f"envs/assets/gs_data")
    maps = {
        "gate_left":"sv_917_3_left_nerfstudio",
        "gate_right":"sv_917_3_right_nerfstudio",
        "simple_hover":"sv_917_3_right_nerfstudio",
        "gate_mid":"sv_1007_gate_mid",
        "clutter":"sv_712_nerfstudio",
        "backroom":"sv_1018_2",
        "flightroom":"sv_1018_3",
    }

    map_folder = os.path.join(gs_path,maps[map])
    print(map_folder)
    for root, _, files in os.walk(map_folder):
        if 'config.yml' in files:
            nerf_cfg_path = os.path.join(root, 'config.yml')

    # Go into NeRF data folder and get NeRF object (because the NeRF instantation
    # requires the current working directory to be the NeRF data folder)
    os.chdir(gs_path)
    gs = GS(Path(nerf_cfg_path), res = resolution)
    os.chdir(main_dir_path)

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
    # # Convert quaternions to rotation matrices in batch
    # rotation_matrices = torch.tensor(
    #     [R.from_quat(quaternion.cpu().numpy()).as_matrix() for quaternion in quaternions], 
    #     device=device
    # )  # Shape: (n, 3, 3)
    
    T_d2f = torch.eye(4, device=device).expand(batch_size, -1, -1).clone()  # Initialize T_d2f
    T_d2f[:, :3, :3] = rotation_matrices  # Apply rotation
    T_d2f[:, :3, 3] = positions  # Apply translation

    # Combine all transformations
    T_c2n_batch = T_f2n @ T_d2f @ T_r2d @ T_c2r  # Shape: (n, 4, 4)

    return T_c2n_batch

def save_rendered_image(image_tensor, save_path="rendered_image.png"):
    """
    Save a rendered image (tensor) to a file.

    Args:
        image_tensor (torch.Tensor): The rendered image (H, W, 3).
        save_path (str): Path to save the image.
    """
    # Convert tensor to NumPy
    image_np = image_tensor.squeeze().cpu().numpy()  # Shape (H, W, 3)
    
    # Ensure it's in [0, 1] range (assuming it's in [0, 255], divide by 255 if needed)
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    # Save using Matplotlib
    plt.imshow(image_np)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    print(f"Image saved to {save_path}")

if __name__ == '__main__':
    gs_dir = Path("/home/david/DiffRL_NeRF/envs/assets/gs_data")
    map_name = 'gate_mid'
    resolution_quality = 1.0

    gs = get_gs(map_name, gs_dir, resolution_quality)

    # Path to saved trajectory files
    real_traj_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_8/gate_mid/02-16-2025-04-02-34_paper_candiate'

    # Define waypoints (must be tensor!)
    waypoints = torch.tensor([
        [-0.2, -0.1, 1.4],
        [1.6, 0.7, 1.1],
        [3.7, 1.35, 0.7],
        [5.8, 0, 1.2]
    ], device=gs.device)

    # Render scene with trajectory and waypoints
    # gs_pose = torch.tensor([[8.3, -0.6, -2.3,  0.0,  0.0,  0.0,  0.17, 0.0, 0.984, 0.0]], device=gs.device)
    gs_pose = torch.tensor([[0.0, 0.0, -1.3,  0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 1.0]], device=gs.device)
    _, img_with_traj = gs.render_with_trajectory(gs_pose, real_traj_path, waypoints)
    save_rendered_image(img_with_traj, "trajectory_tube_waypoints_render.png")

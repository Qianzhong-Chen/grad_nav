import torch
import open3d as o3d
import numpy as np

class ObstacleDistanceCalculator:
    def __init__(self, ply_file, fov_degrees=120.0, device='cuda'):
        """
        Initialize the ObstacleDistanceCalculator with a point cloud and FOV.

        Parameters:
            ply_file (str): Path to the .ply file.
            fov_degrees (float): Field of view in degrees (total angle).
            device (str): Device to load tensors onto ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.points = self.load_point_cloud(ply_file)
        self.fov_degrees = fov_degrees

    def load_point_cloud(self, ply_file):
        """
        Load point cloud data from a .ply file into a PyTorch tensor on the specified device.

        Returns:
            torch.Tensor: Tensor of point cloud coordinates [N, 3].
        """
        pcd = o3d.io.read_point_cloud(ply_file)
        points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=self.device)
        return points

    def quaternion_to_forward_vector(self, quaternions):
        """
        Convert a batch of quaternions to forward direction vectors.

        Parameters:
            quaternions (torch.Tensor): Tensor of quaternions [B, 4], with (w, x, y, z).

        Returns:
            torch.Tensor: Tensor of forward direction vectors [B, 3].
        """
        quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
        q_w = quaternions[:, 0]  
        q_x = quaternions[:, 1]
        q_y = quaternions[:, 2]
        q_z = quaternions[:, 3]

        v_x = 1 - 2 * (q_y ** 2 + q_z ** 2)
        v_y = 2 * (q_x * q_y + q_w * q_z)
        v_z = 2 * (q_x * q_z - q_w * q_y)
        forward_vectors = torch.stack((v_x, v_y, v_z), dim=1) 

        return forward_vectors

    def filter_points_in_fov(self, positions, forward_directions):
        """
        Filter points within the field of view (FOV) from the given positions and forward directions.

        Parameters:
            positions (torch.Tensor): Tensor of observer positions [B, 3].
            forward_directions (torch.Tensor): Tensor of observer forward directions [B, 3].

        Returns:
            list of torch.Tensor: List of filtered points for each batch element.
        """
        forward_directions = forward_directions / forward_directions.norm(dim=1, keepdim=True)
        vectors = self.points.unsqueeze(0) - positions.unsqueeze(1)  
        vectors_norm = vectors / vectors.norm(dim=2, keepdim=True)  
        dot_products = torch.sum(vectors_norm * forward_directions.unsqueeze(1), dim=2)  
        angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))  
        fov_radians = torch.deg2rad(torch.tensor(self.fov_degrees / 2.0, device=self.device))
        mask = angles <= fov_radians 
        filtered_points_list = []
        for b in range(positions.size(0)):
            filtered_points = self.points[mask[b]]  
            filtered_points_list.append(filtered_points)
        return filtered_points_list

    def find_nearest_distance_batch(self, positions, filtered_points_list):
        """
        Find the nearest distance from each position to its corresponding filtered points.

        Parameters:
            positions (torch.Tensor): Tensor of positions [B, 3].
            filtered_points_list (list of torch.Tensor): List of filtered points for each batch element.

        Returns:
            torch.Tensor: Tensor of nearest distances [B].
        """
        distances = []
        for b in range(positions.size(0)):
            filtered_points = filtered_points_list[b]  
            if filtered_points.size(0) == 0:
                distances.append(torch.tensor(float('inf'), device=self.device))
            else:
                diff = filtered_points - positions[b]  
                dist = torch.norm(diff, dim=1)  
                min_dist = dist.min()
                distances.append(min_dist)
        distances = torch.stack(distances)  
        return distances

    def compute_nearest_distances(self, positions, quaternions):
        """
        Compute the nearest distances from positions to obstacles within the FOV using quaternions.

        Parameters:
            positions (torch.Tensor): Tensor of observer positions [B, 3].
            quaternions (torch.Tensor): Tensor of observer quaternions [B, 4], (w, x, y, z).

        Returns:
            torch.Tensor: Tensor of nearest distances [B].
        """
        positions = positions.to(self.device)
        quaternions = quaternions.to(self.device)
        forward_directions = self.quaternion_to_forward_vector(quaternions)  # [B, 3]
        filtered_points_list = self.filter_points_in_fov(positions, forward_directions)
        distances = self.find_nearest_distance_batch(positions, filtered_points_list)
        return distances




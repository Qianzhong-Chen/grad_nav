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
        # Normalize quaternions to ensure they represent valid rotations
        quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)

        # Extract components
        q_w = quaternions[:, 0]  # [B]
        q_x = quaternions[:, 1]
        q_y = quaternions[:, 2]
        q_z = quaternions[:, 3]

        # Compute forward vector components
        # Assuming default forward vector is [1, 0, 0]
        v_x = 1 - 2 * (q_y ** 2 + q_z ** 2)
        v_y = 2 * (q_x * q_y + q_w * q_z)
        v_z = 2 * (q_x * q_z - q_w * q_y)

        forward_vectors = torch.stack((v_x, v_y, v_z), dim=1)  # [B, 3]

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
        # Normalize the forward directions
        forward_directions = forward_directions / forward_directions.norm(dim=1, keepdim=True)  # [B, 3]

        # Vectors from positions to each point
        # positions: [B, 1, 3], points: [1, N, 3]
        vectors = self.points.unsqueeze(0) - positions.unsqueeze(1)  # [B, N, 3]

        # Normalize the vectors
        vectors_norm = vectors / vectors.norm(dim=2, keepdim=True)  # [B, N, 3]

        # Compute the dot product between vectors_norm and forward_directions
        dot_products = torch.sum(vectors_norm * forward_directions.unsqueeze(1), dim=2)  # [B, N]

        # Compute the angle between vectors (in radians)
        angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))  # [B, N]

        # Convert FOV to radians
        fov_radians = torch.deg2rad(torch.tensor(self.fov_degrees / 2.0, device=self.device))

        # Filter points within the FOV
        mask = angles <= fov_radians  # [B, N]

        # For each batch, get the filtered points
        filtered_points_list = []
        for b in range(positions.size(0)):
            filtered_points = self.points[mask[b]]  # [num_filtered_points, 3]
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
            filtered_points = filtered_points_list[b]  # [num_points, 3]
            if filtered_points.size(0) == 0:
                # No points within FOV, set distance to infinity
                distances.append(torch.tensor(float('inf'), device=self.device))
            else:
                # Compute distances from position[b] to filtered_points
                diff = filtered_points - positions[b]  # [num_points, 3]
                dist = torch.norm(diff, dim=1)  # [num_points]
                min_dist = dist.min()
                distances.append(min_dist)
        distances = torch.stack(distances)  # [B]
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
        # Ensure tensors are on the correct device
        positions = positions.to(self.device)
        quaternions = quaternions.to(self.device)

        # Convert quaternions to forward direction vectors
        forward_directions = self.quaternion_to_forward_vector(quaternions)  # [B, 3]

        # Filter points within the field of view
        filtered_points_list = self.filter_points_in_fov(positions, forward_directions)

        # Find nearest distances
        distances = self.find_nearest_distance_batch(positions, filtered_points_list)
        return distances



if __name__ == '__main__':
    positions = torch.tensor([
        [1.0, 2.0, 3.0],  # First position
        [4.0, 5.0, 6.0],  # Second position
    ], device = 'cuda')
    rotation = torch.tensor([
        [0.9239, 0.0, 0.3827, 0.0],  # Quaternion for first position
        [1.0, 0.0, 0.0, 0.0],        # Quaternion for second position (no rotation)
    ])  # [B, 4]

    obst_dist = ObstacleDistanceCalculator(ply_file='712_output_cc.ply')
    distances = obst_dist.compute_nearest_distances(positions, rotation)

    print(distances)

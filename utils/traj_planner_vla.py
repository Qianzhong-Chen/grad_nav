import numpy as np
import open3d as o3d
import heapq
from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
import torch
import time
from scipy.interpolate import interp1d
import pdb


class TrajectoryPlanner:
    def __init__(self, ply_file, voxel_size=0.1, safety_distance=0.2, batch_size=1, wp_distance=2.0, verbose=True):
        self.verbose = verbose
        self.ply_file = ply_file
        self.voxel_size = voxel_size
        self.safety_distance = safety_distance
        self.batch_size = batch_size
        self.points = self.load_point_cloud()
        self.occupancy_grid, self.min_bound = self.create_occupancy_grid()
        self.trajectory_batches = []
        self.waypoints_list = None  # To store per-trajectory waypoints for visualization
        self.destination_positions = None  # To store destinations for visualization
        self.wp_distance = wp_distance
        

    def load_point_cloud(self):
        pcd = o3d.io.read_point_cloud(str(self.ply_file))
        if self.verbose:
            print("Point cloud loaded with {} points.".format(len(pcd.points)))
        return np.asarray(pcd.points)

    def create_occupancy_grid(self):
        min_bound = self.points.min(axis=0) - self.safety_distance
        max_bound = self.points.max(axis=0) + self.safety_distance
        grid_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
        occupancy_grid = np.zeros(grid_shape, dtype=bool)

        indices = np.floor((self.points - min_bound) / self.voxel_size).astype(int)
        occupancy_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # Inflate obstacles by safety distance
        occupancy_grid = binary_dilation(
            occupancy_grid, iterations=int(self.safety_distance / self.voxel_size)
        )
        if self.verbose:
            print("Occupancy grid created with shape {}.".format(occupancy_grid.shape))
        return occupancy_grid, min_bound

    def heuristic(self, a, b):
        return np.linalg.norm(a - b)

    def astar(self, start, goal):
        start_idx = np.floor((start - self.min_bound) / self.voxel_size).astype(int)
        goal_idx = np.floor((goal - self.min_bound) / self.voxel_size).astype(int)

        grid_shape = self.occupancy_grid.shape
        visited = np.full(grid_shape, False, dtype=bool)
        came_from = {}
        g_score = np.full(grid_shape, np.inf)
        g_score[tuple(start_idx)] = 0
        f_score = np.full(grid_shape, np.inf)
        f_score[tuple(start_idx)] = self.heuristic(start_idx, goal_idx)

        open_set = []
        heapq.heappush(open_set, (f_score[tuple(start_idx)], tuple(start_idx)))

        neighbors = [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
        ]

        while open_set:
            current = heapq.heappop(open_set)[1]
            if np.array_equal(current, goal_idx):
                path = []
                while current in came_from:
                    path.append(
                        np.array(current) * self.voxel_size + self.min_bound
                    )
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            visited[current] = True
            for neighbor in neighbors:
                neighbor_idx = np.array(current) + neighbor
                if (
                    0 <= neighbor_idx[0] < grid_shape[0]
                    and 0 <= neighbor_idx[1] < grid_shape[1]
                    and 0 <= neighbor_idx[2] < grid_shape[2]
                ):
                    if (
                        self.occupancy_grid[tuple(neighbor_idx)]
                        or visited[tuple(neighbor_idx)]
                    ):
                        continue
                    tentative_g_score = g_score[current] + self.heuristic(
                        np.array(current), neighbor_idx
                    )
                    if tentative_g_score < g_score[tuple(neighbor_idx)]:
                        came_from[tuple(neighbor_idx)] = current
                        g_score[tuple(neighbor_idx)] = tentative_g_score
                        f_score[tuple(neighbor_idx)] = tentative_g_score + self.heuristic(
                            neighbor_idx, goal_idx
                        )
                        heapq.heappush(
                            open_set, (f_score[tuple(neighbor_idx)], tuple(neighbor_idx))
                        )
        return None

    def resample_trajectory_with_waypoints(self, full_path, waypoints, step_size=2.0):
        trajectory_points = np.array(full_path)
        resampled_trajectory = []

        waypoint_indices = []
        # Find indices of waypoints in the trajectory
        for wp in waypoints:
            distances = np.linalg.norm(trajectory_points - wp, axis=1)
            idx = np.argmin(distances)
            waypoint_indices.append(idx)

        # Sort waypoints indices to ensure order
        waypoint_indices = sorted(set(waypoint_indices))

        # Resample between waypoints
        for i in range(len(waypoint_indices) - 1):
            start_idx = waypoint_indices[i]
            end_idx = waypoint_indices[i + 1]

            segment = trajectory_points[start_idx:end_idx + 1]

            # Compute cumulative distances along the segment
            segment_lengths = np.linalg.norm(np.diff(segment, axis=0), axis=1)
            cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
            total_length = cumulative_lengths[-1]
            num_samples = max(int(np.ceil(total_length / step_size)), 1) + 1
            sample_distances = np.linspace(0, total_length, num_samples)

            # Interpolate the segment at the sample distances
            resampled_segment = np.zeros((num_samples, 3))
            for j in range(3):
                resampled_segment[:, j] = np.interp(
                    sample_distances, cumulative_lengths, segment[:, j]
                )

            # Remove the last point to avoid duplicates except for the final segment
            if i < len(waypoint_indices) - 2:
                resampled_segment = resampled_segment[:-1]

            resampled_trajectory.append(resampled_segment)

        # Concatenate all resampled segments
        resampled_trajectory = np.vstack(resampled_trajectory)
        return resampled_trajectory

    def ensure_increasing_x_with_waypoints(self, trajectory_points, waypoints):
        # Ensure that x is only increasing
        x = trajectory_points[:, 0]
        dx = np.diff(x)
        # Identify indices where x is not decreasing
        valid_indices = np.where(dx >= 0)[0] + 1  # +1 to correct index shift
        valid_indices = np.insert(valid_indices, 0, 0)  # Include the first point

        # Ensure waypoints are included
        waypoint_indices = []
        for wp in waypoints:
            distances = np.linalg.norm(trajectory_points - wp, axis=1)
            idx = np.argmin(distances)
            waypoint_indices.append(idx)
        waypoint_indices = set(waypoint_indices)

        # Combine valid indices and waypoint indices
        combined_indices = sorted(set(valid_indices) | waypoint_indices)

        # Filter the trajectory
        filtered_trajectory = trajectory_points[combined_indices]
        return filtered_trajectory


    def plan_single_trajectory(self, idx, current_pos, destination_pos, waypoints):
        if self.verbose:
            print(f"Planning trajectory {idx+1}/{self.batch_size}...")
        full_path = []
        waypoints = [current_pos] + list(waypoints) + [destination_pos]
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            if self.verbose:
                print(f"Trajectory {idx+1}: Path from {start} to {goal}...")
            path = self.astar(start, goal)
            if path is None:
                if self.verbose:
                    print(f"Trajectory {idx+1}: Path from {start} to {goal} not found.")
                return None
            if full_path and np.array_equal(full_path[-1], path[0]):
                full_path.extend(path[1:])
            else:
                full_path.extend(path)
        if self.verbose:
            print(f"Trajectory {idx+1} planning completed.")
        
        # Convert the full path to a NumPy array
        trajectory_points = np.array(full_path)
        
        # Resample the trajectory between waypoints to ensure waypoints are included
        resampled_trajectory = self.resample_trajectory_with_waypoints(full_path, waypoints, step_size=self.wp_distance)

        # Ensure that x is only increasing, but preserve waypoints
        resampled_trajectory = self.ensure_increasing_x_with_waypoints(resampled_trajectory, waypoints)
        
        return resampled_trajectory.tolist()

    def plan_trajectories(self, current_positions, waypoints_list, des_wp_num=20):
        # Handle data type conversion inside the method
        if isinstance(current_positions, torch.Tensor):
            current_positions = current_positions.clone().detach().cpu().numpy()
        # if isinstance(destination_positions, torch.Tensor):
        #     destination_positions = destination_positions.clone().detach().cpu().numpy()
        if isinstance(waypoints_list, list):
            converted_waypoints_list = []
            for waypoints in waypoints_list:
                if isinstance(waypoints, torch.Tensor):
                    waypoints = waypoints.clone().detach().cpu().numpy()
                converted_waypoints_list.append(waypoints)
            self.waypoints_list = converted_waypoints_list  # Store for visualization
        else:
            raise ValueError("waypoints_list must be a list of torch.tensor or numpy arrays.")

        # Ensure current_positions and destination_positions are NumPy arrays
        current_positions = np.asarray(current_positions)
        destination_positions = np.asarray(waypoints_list[0][-1].unsqueeze(0).clone().detach().cpu().numpy())

        # pdb.set_trace()
        if not (len(current_positions) == len(destination_positions) == len(self.waypoints_list) == self.batch_size):
            raise ValueError("Input lists must have the same length as batch_size.")

        # Prepare arguments for each trajectory
        args_list = []
        for i in range(self.batch_size):
            current_pos = current_positions[i]
            destination_pos = destination_positions[i]
            waypoints = self.waypoints_list[i]

            # Ensure that current_pos and destination_pos are NumPy arrays of shape (3,)
            current_pos = np.asarray(current_pos, dtype=np.float64)
            destination_pos = np.asarray(destination_pos, dtype=np.float64)

            args_list.append((i, current_pos, destination_pos, waypoints))

        # Store destination_positions for visualization
        self.destination_positions = destination_positions

        self.trajectory_batches = [None] * self.batch_size  # Initialize with None

        for i in range(self.batch_size):
            current_pos = current_positions[i]
            destination_pos = destination_positions[i]
            waypoints = self.waypoints_list[i]

            try:
                trajectory = self.plan_single_trajectory(i, current_pos, destination_pos, waypoints)

                # Ensure it's a NumPy array
                trajectory = np.array(trajectory)

                n_points = trajectory.shape[0]
                if n_points < 2:
                    raise ValueError("Trajectory has too few points to interpolate.")

                # Interpolate
                t_original = np.linspace(0, 1, n_points)
                t_new = np.linspace(0, 1, des_wp_num)
                interp_func = interp1d(t_original, trajectory, axis=0)
                resampled_trajectory = interp_func(t_new)

                self.trajectory_batches[i] = resampled_trajectory.tolist()
            except Exception as e:
                print(f"trajectory for env {i} planning failed for {e}")


        # pdb.set_trace()
        return self.trajectory_batches
    
    def visualize_trajectories(self):
        if not self.trajectory_batches:
            if self.verbose:
                print("No trajectories to visualize. Please run plan_trajectories first.")
            return

        def create_tube(trajectory_points, radius, color):
            """
            Creates a tube (cylinder segments) connecting trajectory points.
            """
            tube_mesh = o3d.geometry.TriangleMesh()
            for i in range(len(trajectory_points) - 1):
                start = trajectory_points[i]
                end = trajectory_points[i + 1]
                direction = end - start
                length = np.linalg.norm(direction)
                if length == 0:
                    continue
                direction /= length  # Normalize direction

                # Create cylinder
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)

                # Calculate rotation to align with direction
                z_axis = np.array([0, 0, 1])  # Default cylinder orientation
                rotation_axis = np.cross(z_axis, direction)
                rotation_angle = np.arccos(np.dot(z_axis, direction))
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize rotation axis
                    cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle), center=(0, 0, 0))

                # Translate cylinder to the middle point between start and end
                cylinder.translate((start + end) / 2)

                # Color the cylinder
                cylinder.paint_uniform_color(color)

                tube_mesh += cylinder
            return tube_mesh

        
        # Create geometries list
        geometries = []

        # Add PointCloud (obstacles)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        geometries.append(pcd)

        # Define colors for trajectories
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]

        # Add trajectories as tubes
        for idx, trajectory in enumerate(self.trajectory_batches):
            if trajectory is None or len(trajectory) == 0:
                if self.verbose:
                    print(f"Trajectory {idx+1} is empty or not found.")
                continue

            trajectory_points = np.array(trajectory, dtype=np.float64)
            if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
                if self.verbose:
                    print(f"Trajectory {idx+1} has incorrect shape: {trajectory_points.shape}")
                continue

            color = colors[idx % len(colors)]
            tube_mesh = create_tube(trajectory_points, radius=0.025, color=colors[1])
            geometries.append(tube_mesh)

        # Add waypoints for this trajectory
        waypoints = self.waypoints_list[idx]
        waypoint_spheres = []
        for waypoint in waypoints:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(waypoint)
            sphere.paint_uniform_color(colors[2])  # Same color as trajectory
            waypoint_spheres.append(sphere)
        geometries.extend(waypoint_spheres)

        # Add destination positions to the visualization
        if self.destination_positions is not None:
            destination_spheres = []
            for idx, dest in enumerate(self.destination_positions):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
                sphere.translate(dest)
                color = colors[0]
                sphere.paint_uniform_color(color)  # Same color as trajectory
                destination_spheres.append(sphere)
            geometries.extend(destination_spheres)


        # Visualize all geometries
        o3d.visualization.draw_geometries(geometries)

   
    
# Example usage
if __name__ == "__main__":
    # gate_left
    ply_file = "/home/david/DiffRL_NeRF/envs/assets/point_cloud/sv_917_3_left_nerfstudio.ply"
    batch_size = 1
    current_positions_torch = torch.tensor([[-6.0, -0.30, 1.2]], device='cuda:0').repeat(batch_size,1)
    destination_positions_torch = torch.tensor([[7.0, -2., 1.2]], device='cuda:0').repeat(batch_size,1)

    ##############################################################################
    # Base tasks library
    # "[Task 1] GO THROUGH gate"
    waypoints_torch = [
                torch.tensor([
                                [-4.0, 0.7, 1.3],
                                [-0.2, 0.8, 1.4],
                                [1.0, 0.8, 1.4],
                              ], device='cuda:0'),
                ]*batch_size
    
    # # [Task 2] BYPASS gate (right)
    waypoints_torch = [
                torch.tensor([
                                [-3.0, -0.0, 1.3],
                                [0.0, -0.6, 1.4],
                                [2.0, -0.6, 1.4],
                              ], device='cuda:0'),
                ]*batch_size
    
    # # [Task 3] BYPASS gate (left)
    waypoints_torch = [
                torch.tensor([
                              [-3.0, 0.9, 1.3],
                              [-0.5, 2.0, 1.4],
                              [1.5, 1.5, 1.4],
                              ], device='cuda:0'),
                ]*batch_size
    
    # # [Task 4] FLY ABOVE gate
    waypoints_torch = [
            torch.tensor([
                [-2.0, 0.7, 2.1],
                [-0.5, 0.8, 2.4],
                [1.5, 0.8, 1.6],
                ], device='cuda:0')
            ]*batch_size

    

    planner = TrajectoryPlanner(ply_file, batch_size=batch_size, safety_distance=0.15, wp_distance=2.0, verbose=False)
    start_t = time.time()
    trajectories = planner.plan_trajectories(current_positions_torch, waypoints_torch)
    elapsed_t = time.time() - start_t
    print(f'elapsed time: {elapsed_t}')
    if trajectories:
        wp_list = []
        for i in range(batch_size):
            if trajectories[i] == None:
                wp_list.append(None)
            else:
                wp_list.append(trajectories[i][-1])
        print(wp_list)
        print(len(trajectories[0]))
        print(torch.tensor(trajectories[0]))
        print(trajectories)
        print(len(trajectories[0]))
        planner.visualize_trajectories()

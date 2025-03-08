import numpy as np
import open3d as o3d
import heapq
from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
import torch
import time

class TrajectoryPlanner:
    def __init__(self, ply_file, voxel_size=0.1, safety_distance=0.2, batch_size=1, verbose=True):
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
        

    def load_point_cloud(self):
        pcd = o3d.io.read_point_cloud(self.ply_file)
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

    def plan_single_trajectory(self, args):
        idx, current_pos, destination_pos, waypoints = args
        if self.verbose:
            print(f"Planning trajectory {idx+1}/{self.batch_size}...")

        # Combine waypoints and destination into targets
        targets = list(waypoints) + [destination_pos]

        # Find the next target
        if len(targets) == 0:
            if self.verbose:
                print(f"No waypoints or destination provided for trajectory {idx+1}.")
            return None

        next_target = targets[0]

        # Compute distance to next target
        distance_to_target = np.linalg.norm(next_target - current_pos)
        distance_to_destination = np.linalg.norm(next_target - destination_pos)

        # Always plan wp 2m from curr
        # direction = (next_target - current_pos) / distance_to_target
        # goal = current_pos + direction * 2.0
        # if self.verbose:
        #     print(f"Trajectory {idx+1}: Planning 2 meters ahead towards next target.")

        # Determine goal point
        if distance_to_destination <= 2.0:
            goal = destination_pos
            if self.verbose:
                print(f"Trajectory {idx+1}: Next target within 2 meters.")
        else:
            # Compute point 2 meters ahead along the line to next target
            direction = (next_target - current_pos) / distance_to_target
            goal = current_pos + direction * 2.0
            if self.verbose:
                print(f"Trajectory {idx+1}: Planning 2 meters ahead towards next target.")

        # # Determine goal point
        # if distance_to_target <= 2.0:
        #     goal = next_target
        #     if self.verbose:
        #         print(f"Trajectory {idx+1}: Next target within 2 meters.")
        # else:
        #     # Compute point 2 meters ahead along the line to next target
        #     direction = (next_target - current_pos) / distance_to_target
        #     goal = current_pos + direction * 2.0
        #     if self.verbose:
        #         print(f"Trajectory {idx+1}: Planning 2 meters ahead towards next target.")

        # Plan path from current_pos to goal
        start = current_pos
        if self.verbose:
            print(f"Trajectory {idx+1}: Path from {start} to {goal}...")
        path = self.astar(start, goal)
        if path is None:
            if self.verbose:
                print(f"Trajectory {idx+1}: Path from {start} to {goal} not found.")
            return None

        if self.verbose:
            print(f"Trajectory {idx+1} planning completed.")
        return path

    def plan_trajectories(self, current_positions, destination_positions, waypoints_list):
        # Handle data type conversion inside the method
        if isinstance(current_positions, torch.Tensor):
            current_positions = current_positions.clone().detach().cpu().numpy()
        if isinstance(destination_positions, torch.Tensor):
            destination_positions = destination_positions.clone().detach().cpu().numpy()
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
        destination_positions = np.asarray(destination_positions)

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

        # sequential implementation
        for i in range(self.batch_size):
            current_pos = current_positions[i]
            destination_pos = destination_positions[i]
            waypoints = self.waypoints_list[i]
            try:
                result = self.plan_single_trajectory((i, current_pos, destination_pos, waypoints))
                self.trajectory_batches[i] = result
            except Exception as e:
                print(f'trajectory for env {i} planning failed for {e}')
        # # multi process implementation
        # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        #     futures = []
        #     for args in args_list:
        #         future = executor.submit(self.plan_single_trajectory, args)
        #         futures.append((args[0], future))  # args[0] is the index

        #     for idx, future in futures:
        #         try:
        #             result = future.result(timeout=5)  # Wait up to 5 seconds
        #             self.trajectory_batches[idx] = result
        #         except TimeoutError:
        #             if self.verbose:
        #                 print(f"Trajectory {idx+1} planning timed out after 5 seconds.")
        #             self.trajectory_batches[idx] = None
        #         except Exception as e:
        #             if self.verbose:
        #                 print(f"Trajectory {idx+1} planning failed with exception: {e}")
        #             self.trajectory_batches[idx] = None

        return self.trajectory_batches

    def visualize_trajectories(self):
        if not self.trajectory_batches:
            if self.verbose:
                print("No trajectories to visualize. Please run plan_trajectories first.")
            return

        # Create Open3D PointCloud for the obstacles
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        geometries = [pcd]
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]

        for idx, trajectory in enumerate(self.trajectory_batches):
            if trajectory is None or len(trajectory) == 0:
                if self.verbose:
                    print(f"Trajectory {idx+1} is empty or not found.")
                continue
            # Ensure trajectory_points is a NumPy array of shape (N, 3)
            trajectory_points = np.array(trajectory, dtype=np.float64)
            if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
                if self.verbose:
                    print(f"Trajectory {idx+1} has incorrect shape: {trajectory_points.shape}")
                continue
            if self.verbose:
                print(f"Trajectory {idx+1} points shape: {trajectory_points.shape}")
                print(f"Trajectory {idx+1} points dtype: {trajectory_points.dtype}")

            lines = [[i, i + 1] for i in range(len(trajectory_points) - 1)]
            color = colors[idx % len(colors)]  # Cycle through colors
            line_colors = [color for _ in lines]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(trajectory_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(line_colors)
            geometries.append(line_set)

            # Add waypoints for this trajectory
            waypoints = self.waypoints_list[idx]
            waypoint_spheres = []
            for waypoint in waypoints:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                sphere.translate(waypoint)
                sphere.paint_uniform_color(color)  # Same color as trajectory
                waypoint_spheres.append(sphere)
            geometries.extend(waypoint_spheres)

        # Add destination positions to the visualization
        if self.destination_positions is not None:
            destination_spheres = []
            for idx, dest in enumerate(self.destination_positions):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                sphere.translate(dest)
                color = colors[idx % len(colors)]
                sphere.paint_uniform_color(color)  # Same color as trajectory
                destination_spheres.append(sphere)
            geometries.extend(destination_spheres)

        # Visualize all geometries
        o3d.visualization.draw_geometries(geometries)

# Example usage
if __name__ == "__main__":
    ply_file = "/home/david/DiffRL_NeRF/envs/assets/point_cloud/sv_1007_gate_mid.ply"

    # batch_size = 2
    # current_positions_torch = torch.tensor([[-5.0, 0.0, 1.0], [-3.0, -3.0, 1.0]], device='cuda')
    # destination_positions_torch = torch.tensor([[5.5, -2.3, 1.0], [5.5, -2.3, 1.0]], device='cuda')
    # waypoints_torch = [
    #             torch.tensor([[-0.5, -0.1, 1.4],[3.2, 1.2, 0.6]], device='cuda'),
    #             torch.tensor([[-0.5, -0.1, 1.4]], device='cuda'),
    #             ]

    batch_size = 1
    # current_positions_torch = torch.tensor([[-6.0, 0, 1.0]], device='cuda:0').repeat(batch_size,1)
    # current_positions_torch = torch.tensor([[-1.2, -0.1, 1.4]], device='cuda:0').repeat(batch_size,1)
    current_positions_torch = torch.tensor([[3.2, 1.3, 0.6]], device='cuda:0').repeat(batch_size,1)

    destination_positions_torch = torch.tensor([[6.0, -2.3, 1.0]], device='cuda:0').repeat(batch_size,1)
    waypoints_torch = [
                torch.tensor([
                              [-0.2, -0.1, 1.4],
                              [3.4, 1.3, 0.6],
                              [6.0, -2.3, 1.0]], device='cuda:0'),
                
                ]*batch_size

    # waypoints_torch = [
    #             torch.tensor([[3.4, 1.3, 0.6],
    #                           [6.0, -2.3, 1.0]], device='cuda'),
                
    #             ]*batch_size

    planner = TrajectoryPlanner(ply_file, batch_size=batch_size, safety_distance=0.3, verbose=False)
    start_t = time.time()
    trajectories = planner.plan_trajectories(current_positions_torch, destination_positions_torch, waypoints_torch)
    elapsed_t = time.time() - start_t
    print(f'elapsed time: {elapsed_t}')
    if trajectories:
        wp_list = []
        for i in range(batch_size):
            if trajectories[i] == None:
                wp_list.append(None)
            else:
                wp_list.append(trajectories[i][-1])
        # print(wp_list)
        # print(trajectories)
        planner.visualize_trajectories()

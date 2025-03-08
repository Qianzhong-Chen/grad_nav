import open3d as o3d
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

colors = [
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [1, 0, 1],  # Magenta
    [0, 1, 1],  # Cyan
]

# # Gate_mid
# real_traj_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_8/gate_mid/02-16-2025-04-02-34_paper_candiate'
# waypoints_torch = [
#     torch.tensor([
#         [-0.2, -0.1, 1.4],
#         [1.6, 0.7, 1.1],
#         [3.7, 1.35, 0.7],
#         [5.8, 0, 1.2],
#     ], device='cuda:0')
# ] * 1

# Gate_right
real_traj_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_8/gate_right/02-17-2025-00-23-32'
waypoints_torch = [
                torch.tensor([
                              [-3.0, -1.3, 1.3],
                              [0.2, -1.0, 1.4],
                              [1.8, 0.3, 1.1],
                              [3.7, 1.25, 0.7],
                              [5.8, 0, 1.2],
                              ], device='cuda:0'),
                
                ]*1

x = np.load(os.path.join(real_traj_path, "x.npy"), allow_pickle=True) - 6.0
y = np.load(os.path.join(real_traj_path, "y.npy"), allow_pickle=True)
z = np.load(os.path.join(real_traj_path, "z.npy"), allow_pickle=True) - 0.2

assert x.shape == y.shape == z.shape, "Mismatch in array shapes"

n = len(x)
vx = np.gradient(x)
vy = np.gradient(y)
vz = np.gradient(z)

Xref = np.stack([x, y, z, vx, vy, vz])
Xref = Xref.astype(np.float64)

waypoints = waypoints_torch[0].cpu().numpy()  # Convert from torch to numpy

combined_mesh = o3d.geometry.TriangleMesh()

# Function to create a colored cylinder
def create_cylinder(start_point, end_point, color):
    direction = end_point - start_point
    height = np.linalg.norm(direction)

    if height > 0:
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=height)
        cylinder.compute_vertex_normals()

        # Align the cylinder with the line direction
        z_axis = np.array([0, 0, 1])
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction) / height)
        if np.linalg.norm(axis) > 0:
            axis = axis / np.linalg.norm(axis)
            cylinder.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle))

        # Translate the cylinder to the start point
        cylinder.translate(start_point)

        # Ensure per-vertex color assignment
        vertex_colors = np.tile(color, (np.asarray(cylinder.vertices).shape[0], 1))
        cylinder.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        return cylinder
    return None

# Normalize velocity colors
velocities = np.linalg.norm(Xref[3:6, :-1], axis=0)
norm_velocities = (velocities - velocities.min()) / (velocities.max() - velocities.min())
cmap = plt.get_cmap('viridis')

# Add cylinders for trajectory
for i in range(Xref.shape[1] - 1):
    start_point = Xref[:3, i].copy()
    end_point = Xref[:3, i + 1].copy()

    # Apply rotation if needed
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    start_point = rotation_matrix_x @ start_point
    end_point = rotation_matrix_x @ end_point

    # Get color for this segment
    color = cmap(norm_velocities[i])[:3]  # RGB from colormap

    cylinder = create_cylinder(start_point, end_point, color)
    if cylinder:
        combined_mesh += cylinder

# Add waypoints as colored spheres
for waypoint in waypoints[:-1]:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    sphere.translate(waypoint)
    vertex_colors = np.tile(colors[2], (np.asarray(sphere.vertices).shape[0], 1))  # Ensure per-vertex coloring
    sphere.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    combined_mesh += sphere

# Add start point (box)
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
sphere.translate(Xref[:3,0])
vertex_colors = np.tile(colors[4], (np.asarray(sphere.vertices).shape[0], 1))
sphere.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
combined_mesh += sphere

# Add goal point (sphere)
goal_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
goal_sphere.translate(waypoints[-1])
vertex_colors = np.tile(colors[0], (np.asarray(goal_sphere.vertices).shape[0], 1))
goal_sphere.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
combined_mesh += goal_sphere

# Save and visualize
output_path = os.path.join(real_traj_path, 'traj_with_waypoints.obj')
o3d.io.write_triangle_mesh(output_path, combined_mesh)
o3d.visualization.draw_geometries([combined_mesh])

print(f"Saved 3D trajectory with waypoints at: {output_path}")

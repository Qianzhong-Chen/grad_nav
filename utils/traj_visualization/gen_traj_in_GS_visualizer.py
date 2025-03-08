import os
import time
from pathlib import Path
from splat.splat_utils import GSplatLoader
import torch
import numpy as np
import trimesh
import viser
import viser.transforms as tf
import pdb
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors_table = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
]

### PARAMETERS ###
# # gate_mid
# scene_name = 'gate_mid'    
# real_traj_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_multi_gate/gate_mid/02-23-2025-17-07-16_paper_candidate'
# traj_end = 370
# # traj_mesh_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_multi_gate/gate_mid/02-23-2025-17-07-16_paper_candidate/traj_with_waypoints.obj'           
# # Path to Gaussian Splat configuration
# path_to_gsplat = Path('/home/david/DiffRL_NeRF/envs/assets/gs_data/sv_1007_gate_mid/splatfacto/2024-10-07_145741/config.yml')
# waypoints = np.array([
#                             [-2.0, 0.0, 1.3],
#                             [-0.2, -0.2, 1.4],
#                             [2.0, -0.5, 1.3],
#                             ])
                

# # gate_right
# scene_name = 'gate_right'    
# real_traj_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_multi_gate/gate_right/02-25-2025-04-02-33_paper_candidate'
# traj_end = 350
# # traj_mesh_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_multi_gate/gate_right/02-23-2025-16-50-01_paper_candidate/traj_with_waypoints.obj'           
# # Path to Gaussian Splat configuration
# path_to_gsplat = Path('/home/david/DiffRL_NeRF/envs/assets/gs_data/sv_917_3_right_nerfstudio/splatfacto/2024-09-17_202436/config.yml')
# waypoints = np.array([
#                     [-2.0, -1.0, 1.3],
#                     [-0.2, -1.2, 1.4],
#                     [2.0, -1.2, 1.3],
#                                     ])

# gate_left
scene_name = 'gate_left'  
# real_traj_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_multi_gate/gate_left/02-23-2025-17-09-51_paper_candidate' 
real_traj_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_multi_gate/gate_left/03-03-2025-17-31-03' 
traj_end = -1
# traj_mesh_path = '/home/david/DiffRL_NeRF/examples/outputs/quadrotor_gs_mask_pos_traj_global_multi_gate/gate_left/02-23-2025-17-09-51_paper_candidate/traj_with_waypoints.obj'           
# Path to Gaussian Splat configuration
path_to_gsplat = Path('/home/david/DiffRL_NeRF/envs/assets/gs_data/sv_917_3_left_nerfstudio/splatfacto/2024-09-17_192401/config.yml')
waypoints = np.array([
                    [-2.0, 0.7, 1.3],
                    [-0.2, 1.1, 1.4],
                    [1.8, 1.1, 1.2],
                                    ])

scale_factor = 1.0
# Load Gaussian Splat scene
gsplat = GSplatLoader(path_to_gsplat, device)
server = viser.ViserServer()

# Identity rotation
rotation = tf.SO3.from_x_radians(0.0).wxyz

# Filter Gaussian splats (if needed)
mask = torch.ones(gsplat.means.shape[0], dtype=torch.bool, device=device)
means, covs, colors, opacities = gsplat.means[mask], gsplat.covs[mask], gsplat.colors[mask], gsplat.opacities[mask]

# Add Gaussian splats to the scene
server.scene.add_gaussian_splats(
    name="/splats",
    centers=means.cpu().numpy(),
    covariances=covs.cpu().numpy(),
    rgbs=colors.cpu().numpy(),
    opacities=opacities.cpu().numpy(),
    wxyz=rotation,
)

# Load and display Gaussian splat mesh representation
mesh_path = f"assets/{scene_name}.obj"
if not os.path.exists(mesh_path):
    os.makedirs('assets', exist_ok=True)
    gsplat.save_mesh(mesh_path, res=4)


# Line segments
x = np.load(os.path.join(real_traj_path, "x.npy"), allow_pickle=True) - 6.0
y = np.load(os.path.join(real_traj_path, "y.npy"), allow_pickle=True)
z = np.load(os.path.join(real_traj_path, "z.npy"), allow_pickle=True) - 0.2
z = np.ones_like(z) * 1.35

vx = np.gradient(x)
vy = np.gradient(y)
vz = np.gradient(z)
Xref = np.stack([x, y, z, vx, vy, vz])
Xref = Xref.astype(np.float64)
Xref = Xref[:, :traj_end]


points = Xref[:3, :]
points = points.astype(np.float64)
traj = np.array(points.T)     # Nx3
traj = np.stack([traj[:-1], traj[1:]], axis=1)  # Nx2x3

# Normalize velocity colors
velocities = np.linalg.norm(Xref[3:6, :-1], axis=0)
norm_velocities = (velocities - velocities.min()) / (velocities.max() - velocities.min())
norm_velocities = norm_velocities ** 0.5  # Apply power transformation to boost contrast
cmap = plt.get_cmap('jet')  # Use 'jet' colormap for more vivid colors

traj_colors = np.zeros((Xref.shape[1] - 1, 2, 3), dtype=np.float32)
for i in range(Xref.shape[1] - 1):
    color_start = np.array(cmap(norm_velocities[i])[:3])
    color_end = np.array(cmap(norm_velocities[i])[:3])

    traj_colors[i, 0] = color_start
    traj_colors[i, 1] = color_end



# Draw trajectory
server.scene.add_line_segments(
    name="/trajectory",
    points = traj,
    colors = traj_colors,
    # colors = np.tile([0.0,1.0,0], (traj.shape[0], 2, 1)),
    line_width = 20.0
)


# Start waypoint 
server.scene.add_icosphere(
    name="/start_waypoint",
    position=traj[0][0],
    color=colors_table[-2],
    radius=0.05,
    visible=True,
)

# Draw each waypoint as a sphere
for i, waypoint in enumerate(waypoints[:-1]):
    server.scene.add_icosphere(
        name=f"/waypoints_{i}",  # Unique name for each waypoint
        position=waypoint,  # 3D position of the waypoint
        color=colors_table[2],  # Color of the sphere
        radius=0.15,  # Radius of the sphere
        visible=True,
    )

server.scene.add_icosphere(
    name=f"/end_waypoint",  # Unique name for each waypoint
    position=waypoints[-1],  # 3D position of the waypoint
    color=colors_table[0],  # Color of the sphere
    radius=0.35,  # Radius of the sphere
    visible=True,
)

time.sleep(30.0)

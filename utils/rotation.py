import torch

def quat_multiply_batch(q1, q2):
    """Multiplies two batches of quaternions q1 and q2 element-wise."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=1)

# def dynamic_to_nerf_quaternion_batch(dynamic_quat, angle):
#     # The relative rotation is constant for all quaternions in the batch
#     relative_rotation = torch.tensor([0.707, 0.707, 0.0, 0.0], dtype=dynamic_quat.dtype, device=dynamic_quat.device)
    
#     # Expand relative_rotation to match batch size
#     batch_size = dynamic_quat.shape[0]
#     relative_rotation_batch = relative_rotation.unsqueeze(0).expand(batch_size, -1)
    
#     # Apply the relative rotation to each quaternion in the batch
#     return quat_multiply_batch(relative_rotation_batch, dynamic_quat) # (w,x,y,z)

def dynamic_to_nerf_quaternion_batch(dynamic_quat, angle):
    nerf_quat = torch.zeros_like(dynamic_quat,device=dynamic_quat.device)
    nerf_quat[:,0] = dynamic_quat[:,3]
    nerf_quat[:,1] = dynamic_quat[:,0]
    nerf_quat[:,2] = dynamic_quat[:,1]
    nerf_quat[:,3] = dynamic_quat[:,2]

    nerf_quat[:,0] = nerf_quat[:,0] - torch.cos(torch.tensor(angle/2))
    nerf_quat[:,1] = nerf_quat[:,1] - torch.sin(torch.tensor(angle/2))
    return nerf_quat # (w,x,y,z)

def pose_transfer_ns(position, quaternion):
    qw, qx, qy, qz = quaternion
    # deal with special position convention of dFlex 
    x = position[0]; z = position[1]; y = position[2] 
    pos_reframe = torch.tensor([x,y,z])
    R = torch.tensor([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    T = pos_reframe.view(3, 1)
    ns_pose_matrix = torch.cat((R, T), dim=1)
    return ns_pose_matrix

# def pose_transfer_ns_batched(torso_pos, torso_rot):
#     # Split the quaternions
#     qw = torso_rot[:, 0]
#     qx = torso_rot[:, 1]
#     qy = torso_rot[:, 2]
#     qz = torso_rot[:, 3]
#     # Deal with special position convention of dFlex
#     x = torso_pos[:, 0]
#     z = torso_pos[:, 1]
#     y = torso_pos[:, 2]
#     pos_reframe = torch.stack([x, y, z], dim=1)  # Shape: (num_envs, 3)
#     # Compute the rotation matrices in a batched manner
#     R = torch.stack([
#         1 - 2 * (qy**2 + qz**2),
#         2 * (qx * qy - qz * qw),
#         2 * (qx * qz + qy * qw),
#         2 * (qx * qy + qz * qw),
#         1 - 2 * (qx**2 + qz**2),
#         2 * (qy * qz - qx * qw),
#         2 * (qx * qz - qy * qw),
#         2 * (qy * qz + qx * qw),
#         1 - 2 * (qx**2 + qy**2)
#     ], dim=-1).view(-1, 3, 3)  # Shape: (num_envs, 3, 3)
#     # Reshape the position to (num_envs, 3, 1)
#     T = pos_reframe.unsqueeze(-1)  # Shape: (num_envs, 3, 1)
#     # Concatenate rotation and position to form the final pose matrix
#     ns_pose = torch.cat((R, T), dim=-1)  # Shape: (num_envs, 3, 4)

#     return ns_pose

# def pose_transfer_ns_batched(torso_pos, nerf_rot):
#     """
#     Generate a batched n x 3 x 4 transformation matrix from torso positions and NeRF rotations.
    
#     Parameters:
#     - torso_pos: n x 3 torch tensor, where each row represents (x, z, y) position of the torso.
#     - nerf_rot: n x 4 torch tensor, where each row represents a quaternion (w, x, z, y).
    
#     Returns:
#     - n x 3 x 4 torch tensor representing the transformation matrix for each batch.
#     """

#     # Ensure inputs are torch tensors
#     assert torso_pos.shape[1] == 3, "torso_pos should be of shape (n, 3)"
#     assert nerf_rot.shape[1] == 4, "nerf_rot should be of shape (n, 4)"
    
#     # Split quaternion into its components
#     w, x, z, y = nerf_rot[:, 0], nerf_rot[:, 1], nerf_rot[:, 2], nerf_rot[:, 3]
    
#     # Compute the rotation matrix from the quaternion
#     R = torch.zeros((nerf_rot.shape[0], 3, 3), device=nerf_rot.device)
    
#     # Filling rotation matrix
#     R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
#     R[:, 0, 1] = 2 * (x * y - z * w)
#     R[:, 0, 2] = 2 * (x * z + y * w)
    
#     R[:, 1, 0] = 2 * (x * y + z * w)
#     R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
#     R[:, 1, 2] = 2 * (y * z - x * w)
    
#     R[:, 2, 0] = 2 * (x * z - y * w)
#     R[:, 2, 1] = 2 * (y * z + x * w)
#     R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
#     # Create the batched transformation matrix (n x 3 x 4)
#     T = torch.zeros((nerf_rot.shape[0], 3, 4), device=nerf_rot.device)
    
#     # Fill the rotation part
#     T[:, :, :3] = R
    
#     # Correct the order of translation: (x, z, y) -> (x, y, z) for the last column
#     T[:, 0, 3] = torso_pos[:, 0]  # x remains the same
#     T[:, 1, 3] = torso_pos[:, 2]  # y should take the value of z
#     T[:, 2, 3] = torso_pos[:, 1]  # z should take the value of y
    
#     return T

def quaternion_to_euler(quaternion):
    """
    Converts a batch of quaternions in the format (w, x, z, y) to roll, pitch, yaw.
    quaternion: Tensor of shape (batch_size, 4) where each quaternion is (w, x, z, y).
    Returns: Tensor of shape (batch_size, 3) with roll, pitch, yaw angles.
    """
    w, x, z, y = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.copysign(torch.tensor(3.141592653589793 / 2), sinp),  # pi/2 or -pi/2
        torch.asin(sinp)
    )

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Stack the results into a single tensor
    return torch.stack((roll, pitch, yaw), dim=1)

def quaternion_yaw_forward(quat):
    """
    Extracts the forward direction vector (x, y) from a quaternion (x, y, z. w).
    This avoids computing explicit Euler angles, reducing numerical instability.
    
    quat: Tensor of shape (batch_size, 4), quaternion in (x, y, z, w) format.
    Returns: Tensor of shape (batch_size, 2) representing the forward direction (x, y).
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Compute the forward direction in the XY plane (yaw-aligned heading)
    forward_x = 2 * (w * w + x * x) - 1  # Equivalent to cos(yaw)
    forward_y = -2 * (x * y + w * z)      # Equivalent to sin(yaw)

    heading_vec = torch.stack((forward_x, forward_y), dim=1)
    heading_vec = heading_vec / (torch.norm(heading_vec, dim=-1, keepdim=True) + 1e-8)
    
    return heading_vec

def pose_transfer_ns_batched(torso_pos, nerf_rot):
    """
    Generate a batched n x 3 x 4 transformation matrix from torso positions and NeRF rotations.
    
    Parameters:
    - torso_pos: n x 3 torch tensor, where each row represents (x, z, y) position of the torso.
    - nerf_rot: n x 4 torch tensor, where each row represents a quaternion (w, x, z, y).
    
    Returns:
    - n x 3 x 4 torch tensor representing the transformation matrix for each batch.
    """

    # Ensure inputs are torch tensors
    assert torso_pos.shape[1] == 3, "torso_pos should be of shape (n, 3)"
    assert nerf_rot.shape[1] == 4, "nerf_rot should be of shape (n, 4)"
    
    # Split quaternion into its components, based on (w, x, z, y) convention
    w, x, z, y = nerf_rot[:, 0], nerf_rot[:, 1], nerf_rot[:, 2], nerf_rot[:, 3]
    
    # Compute the rotation matrix from the quaternion
    R = torch.zeros((nerf_rot.shape[0], 3, 3), device=nerf_rot.device)
    
    # Corrected rotation matrix components with the appropriate sign for yaw (z-axis)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y + z * w)   # Switched sign for z * w
    R[:, 0, 2] = 2 * (x * z - y * w)
    
    R[:, 1, 0] = 2 * (x * y - z * w)   # Switched sign for z * w
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z + x * w)
    
    R[:, 2, 0] = 2 * (x * z + y * w)
    R[:, 2, 1] = 2 * (y * z - x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    # Create the batched transformation matrix (n x 3 x 4)
    T = torch.zeros((nerf_rot.shape[0], 3, 4), device=nerf_rot.device)
    
    # Fill the rotation part
    T[:, :, :3] = R
    
    # Correct the order of translation: (x, z, y) -> (x, y, z) for the last column
    T[:, 0, 3] = torso_pos[:, 0]  # x remains the same
    T[:, 1, 3] = torso_pos[:, 2]  # y should take the value of z
    T[:, 2, 3] = torso_pos[:, 1]  # z should take the value of y
    
    return T

def rot_mat_2_rpy(pose:torch.tensor):
    '''input tensor size: (1, 3, 4) '''
    rotation_matrix = pose[:, :, :3]

    # Calculate pitch
    pitch = torch.arcsin(-rotation_matrix[0, 2, 0])

    # Calculate yaw
    yaw = torch.atan2(rotation_matrix[0, 1, 0], rotation_matrix[0, 0, 0])

    # Calculate roll
    roll = torch.atan2(rotation_matrix[0, 2, 1], rotation_matrix[0, 2, 2])

    return roll, pitch, yaw

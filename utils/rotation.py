import torch

def quaternion_to_euler(quaternion):
    """
    Converts a batch of quaternions in the format (w, x, z, y) to roll, pitch, yaw.
    quaternion: Tensor of shape (batch_size, 4) where each quaternion is (w, x, z, y).
    Returns: Tensor of shape (batch_size, 3) with roll, pitch, yaw angles.
    """
    w, x, z, y = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.copysign(torch.tensor(3.141592653589793 / 2), sinp),  # pi/2 or -pi/2
        torch.asin(sinp)
    )
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=1)

def quaternion_yaw_forward(quat):
    """
    Extracts the forward direction vector (x, y) from a quaternion (x, y, z. w).
    This avoids computing explicit Euler angles, reducing numerical instability.
    
    quat: Tensor of shape (batch_size, 4), quaternion in (x, y, z, w) format.
    Returns: Tensor of shape (batch_size, 2) representing the forward direction (x, y).
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    forward_x = 2 * (w * w + x * x) - 1  # Equivalent to cos(yaw)
    forward_y = -2 * (x * y + w * z)      # Equivalent to sin(yaw)
    heading_vec = torch.stack((forward_x, forward_y), dim=1)
    heading_vec = heading_vec / (torch.norm(heading_vec, dim=-1, keepdim=True) + 1e-8)
    
    return heading_vec


import torch

class QuadrotorSimulator:
    def __init__(self, mass=1.0, inertia=None, link_length=0.1, Kp=None, Kd=None, 
                 freq=100.0, max_thrust=15.0, total_time=1.0, rotor_noise_std=0.01, br_noise_std=0.01,
                 drag_coeff=0.5, cross_area=0.1, air_density=1.225):
        """
        Initialize the quadrotor simulator with user-defined parameters.

        Args:
            mass (torch.Tensor): Mass of the quadrotors (batch_size,).
            inertia (torch.Tensor): Inertia matrix (batch_size, 3, 3).
            link_length (float): Distance from the center to each rotor.
            Kp (torch.Tensor): Proportional gains for angular rate control (batch_size, 3).
            Kd (torch.Tensor): Derivative gains for angular rate control (batch_size, 3).
            freq (float): Simulation frequency (Hz).
            max_thrust (torch.Tensor): Maximum thrust per rotor (batch_size,).
            total_time (float): Total simulation time (seconds).
            rotor_noise_std (float): Standard deviation of rotor control noise.
            br_noise_std (float): Standard deviation of body rate control noise.
            drag_coeff (float): Drag coefficient.
            cross_area (float): Cross-sectional area for drag calculation.
            air_density (float): Air density.
        """
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Quadrotor dynamics properties
        self.m = mass.to(self.device)  
        if inertia is None:
            self.I = torch.diag_embed(torch.tensor([0.01, 0.01, 0.02], device=self.device)).unsqueeze(0).repeat(mass.shape[0], 1, 1)
        else:
            self.I = inertia.to(self.device)  
        self.I_inv = torch.inverse(self.I)  
        self.l = torch.tensor(link_length, device=self.device) 
        self.max_thrust = max_thrust.to(self.device)  
        # PD gains for angular rate control
        if Kp is None:
            self.Kp = torch.tensor([1.0, 1.0, 1.0], device=self.device).unsqueeze(0).repeat(mass.shape[0], 1)
        else:
            self.Kp = Kp.to(self.device)  
        if Kd is None:
            self.Kd = torch.tensor([0.1, 0.1, 0.1], device=self.device).unsqueeze(0).repeat(mass.shape[0], 1)
        else:
            self.Kd = Kd.to(self.device)  

        # Simulation parameters
        self.dt = torch.tensor(1.0 / freq, device=self.device)
        self.n_steps = int(total_time * freq)
        self.prev_omega = 0.0

        # Gravity vector
        self.g = torch.tensor([0.0, 0.0, -9.81], device=self.device)

        # Noise parameters
        self.rotor_noise_std = rotor_noise_std
        self.br_noise_std = br_noise_std

        # Drag parameters
        self.drag_coeff = drag_coeff
        self.cross_area = cross_area
        self.air_density = air_density


    def quaternion_to_rotation_matrix(self, q):
        """
        Convert a quaternion to a rotation matrix.
        """
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        batch_size = q.shape[0]

        R = torch.zeros((batch_size, 3, 3), device=self.device)

        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z - w*y)

        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - w*x)

        R[:, 2, 0] = 2 * (x*z + w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R

    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions.
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([w, x, y, z], dim=1)

    def integrate_quaternion(self, q, omega, dt):
        """
        Integrate quaternion over time using angular velocity.
        """
        zeros = torch.zeros((omega.shape[0], 1), device=self.device)
        omega_quat = torch.cat((zeros, omega), dim=1)
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
        q_new = q + q_dot * dt
        q_new = q_new / q_new.norm(dim=1, keepdim=True)
        return q_new

    def update(self, position, velocity, orientation, angular_velocity, omega_desired_body, thrust, motor_speeds):
        """
        Update the quadrotor's state.
        """
        batch_size = position.shape[0]

        # Apply rotor control noise
        if self.rotor_noise_std is not None:
            thrust_noise = torch.randn_like(thrust) * self.rotor_noise_std
        actual_thrust = (thrust + thrust_noise) * self.max_thrust.clone()

        # Apply body rate noise
        if self.br_noise_std is not None:
            omega_noise = torch.randn_like(omega_desired_body) * self.br_noise_std
            omega_desired_body = omega_desired_body + omega_noise

        omega_error = omega_desired_body - angular_velocity
        est_alpha = (angular_velocity - self.prev_omega) / self.dt
        est_alpha = est_alpha.detach()
        Tau = self.Kp * omega_error - self.Kd * est_alpha
        omega_cross = torch.cross(angular_velocity.clone(), torch.bmm(self.I.clone().detach(), angular_velocity.clone().unsqueeze(2)).squeeze(2))
        omega_dot = torch.bmm(self.I_inv.clone().detach(), (Tau - omega_cross).unsqueeze(2)).squeeze(2)
        next_angular_velocity = angular_velocity + omega_dot * self.dt
        next_orientation = self.integrate_quaternion(orientation, next_angular_velocity, self.dt)
        R = self.quaternion_to_rotation_matrix(next_orientation)

        F_body = torch.zeros((batch_size, 3), device=self.device)
        F_body[:, 2] = actual_thrust.clone()
        F_world = torch.bmm(R, F_body.unsqueeze(2)).squeeze(2)
        drag_force = -0.5 * self.drag_coeff * self.cross_area * self.air_density * velocity.clone() * torch.norm(velocity.clone(), dim=1, keepdim=True)
        
        linear_acceleration = (F_world + drag_force.clone()) / self.m.clone().unsqueeze(1) + self.g
        next_velocity = velocity + linear_acceleration * self.dt
        next_position = position + next_velocity * self.dt
        self.prev_omega = angular_velocity

        return next_position, next_velocity, next_orientation, next_angular_velocity, linear_acceleration, omega_dot, motor_speeds

    def run_simulation(self, position, velocity, orientation, angular_velocity, control_input):
        """
        Simulate multiple time steps given the current state and control inputs.

        Args:
            position (torch.Tensor): Current position (batch_size, 3).
            velocity (torch.Tensor): Current velocity (batch_size, 3).
            orientation (torch.Tensor): Current orientation quaternion (batch_size, 4).
            angular_velocity (torch.Tensor): Current angular velocity (batch_size, 3).
            control_input (tuple): A tuple of (omega_desired_body, thrust), where:
                - omega_desired_body (torch.Tensor): Desired body rates (batch_size, 3).
                - thrust (torch.Tensor): Normalized thrust (batch_size, 1).

        Returns:
            final_position (torch.Tensor): Final position after simulation (batch_size, 3).
            final_velocity (torch.Tensor): Final velocity after simulation (batch_size, 3).
            final_angular_velocity (torch.Tensor): Final angular velocity after simulation (batch_size, 3).
            final_orientation (torch.Tensor): Final orientation quaternion after simulation (batch_size, 4).
            final_lin_acc (torch.Tensor): Final linear acceleration (batch_size, 3).
            final_ang_acc (torch.Tensor): Final angular acceleration (batch_size, 3).
        """
        # Unpack control inputs
        omega_desired_body, thrust = control_input
        motor_speeds = torch.zeros_like(thrust, device=self.device)

        # Simulation loop
        for step in range(self.n_steps):
            position, velocity, orientation, angular_velocity, linear_acceleration, angular_acceleration, motor_speeds = self.update(
                position, velocity, orientation, angular_velocity, omega_desired_body, thrust, motor_speeds
            )

        # Return final states
        return position, velocity, angular_velocity, orientation, linear_acceleration, angular_acceleration
    

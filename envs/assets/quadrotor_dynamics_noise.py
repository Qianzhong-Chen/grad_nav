import torch
import time

class QuadrotorSimulator:
    def __init__(self, mass=1.0, inertia=None, link_length=0.1, Kp=None, Kd=None, 
                 freq=100.0, max_thrust=15.0, total_time=1.0, rotor_noise_std=None, br_noise_std = None):
        """
        Initialize the quadrotor simulator with user-defined parameters.

        Args:
            mass (float): Mass of the quadrotor.
            inertia (torch.Tensor): Inertia matrix (3x3).
            link_length (float): Distance from the center to each rotor.
            Kp (torch.Tensor): Proportional gains for angular rate control (3,).
            Kd (torch.Tensor): Derivative gains for angular rate control (3,).
            freq (float): Simulation frequency (Hz).
            max_thrust (float): Maximum thrust per rotor.
            total_time (float): Total simulation time (seconds).
            rotor_noise_std (float): Standard deviation of rotor control noise.
        """
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Quadrotor dynamics properties
        self.m = torch.tensor(mass, device=self.device)
        if inertia is None:
            # Default to a simple diagonal inertia matrix
            self.I = torch.diag(torch.tensor([0.01, 0.01, 0.02], device=self.device))
        else:
            self.I = inertia.to(self.device)
        self.I_inv = torch.inverse(self.I)  # Precompute the inverse of the inertia matrix
        self.l = torch.tensor(link_length, device=self.device)  # Link length
        self.max_thrust = torch.tensor(max_thrust, device=self.device)  # Maximum thrust

        # PD gains for angular rate control
        if Kp is None:
            self.Kp = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        else:
            self.Kp = Kp.to(self.device)
        if Kd is None:
            self.Kd = torch.tensor([0.1, 0.1, 0.1], device=self.device)
        else:
            self.Kd = Kd.to(self.device)

        # Simulation parameters
        self.dt = torch.tensor(1.0 / freq, device=self.device)
        self.n_steps = int(total_time * freq)

        # Gravity vector
        self.g = torch.tensor([0.0, 0.0, -9.81], device=self.device)

        # Rotor control noise standard deviation
        self.rotor_noise_std = rotor_noise_std
        self.br_noise_std = br_noise_std # body rate noise 


    # def quaternion_to_rotation_matrix(self, q):
    #     """
    #     Convert a quaternion to a rotation matrix.
    #     """
    #     # q is of shape (batch_size, 4)
    #     w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    #     batch_size = q.shape[0]

    #     # Pre-compute repeated expressions
    #     ww = w * w
    #     xx = x * x
    #     yy = y * y
    #     zz = z * z
    #     wx = w * x
    #     wy = w * y
    #     wz = w * z
    #     xy = x * y
    #     xz = x * z
    #     yz = y * z

    #     # Initialize rotation matrix
    #     R = torch.zeros((batch_size, 3, 3), device=self.device)

    #     R[:, 0, 0] = 1 - 2 * (yy + zz)
    #     R[:, 0, 1] = 2 * (xy - wz)
    #     R[:, 0, 2] = 2 * (xz + wy)

    #     R[:, 1, 0] = 2 * (xy + wz)
    #     R[:, 1, 1] = 1 - 2 * (xx + zz)
    #     R[:, 1, 2] = 2 * (yz - wx)

    #     R[:, 2, 0] = 2 * (xz - wy)
    #     R[:, 2, 1] = 2 * (yz + wx)
    #     R[:, 2, 2] = 1 - 2 * (xx + yy)

    #     return R

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert a quaternion to a rotation matrix with corrected pitch direction.
        """
        # q is of shape (batch_size, 4)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        batch_size = q.shape[0]

        # Compute the rotation matrix elements
        R = torch.zeros((batch_size, 3, 3), device=self.device)

        # Corrected signs for pitch (y-axis rotation)
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z - w*y)  # Changed sign: "+" -> "-"

        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - w*x)

        R[:, 2, 0] = 2 * (x*z + w*y)  # Changed sign: "-" -> "+"
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R

    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions.
        """
        # q1 and q2 are of shape (batch_size, 4)
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

        Args:
            q (torch.Tensor): Orientation quaternion (batch_size, 4)
            omega (torch.Tensor): Angular velocity (batch_size, 3)
            dt (torch.Tensor): Time step (scalar)
        Returns:
            q_new (torch.Tensor): Updated quaternion (batch_size, 4)
        """
        # Angular velocity quaternion
        zeros = torch.zeros((omega.shape[0], 1), device=self.device)
        omega_quat = torch.cat((zeros, omega), dim=1)
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
        q_new = q + q_dot * dt
        q_new = q_new / q_new.norm(dim=1, keepdim=True)
        return q_new

    def update(self, position, velocity, orientation, angular_velocity, omega_desired, thrust):
        """
        Update the quadrotor's state based on desired body rates and thrust.

        Args:
            position (torch.Tensor): Current position (batch_size, 3)
            velocity (torch.Tensor): Current velocity (batch_size, 3)
            orientation (torch.Tensor): Current orientation quaternion (batch_size, 4)
            angular_velocity (torch.Tensor): Current angular velocity (batch_size, 3)
            omega_desired (torch.Tensor): Desired body rates (batch_size, 3)
            thrust (torch.Tensor): Normalized thrust (batch_size, 1)

        Returns:
            next_position (torch.Tensor): Next position (batch_size, 3)
            next_velocity (torch.Tensor): Next velocity (batch_size, 3)
            next_orientation (torch.Tensor): Next orientation quaternion (batch_size, 4)
            next_angular_velocity (torch.Tensor): Next angular velocity (batch_size, 3)
            linear_acceleration (torch.Tensor): Linear acceleration (batch_size, 3)
            angular_acceleration (torch.Tensor): Angular acceleration (batch_size, 3)
        """
        batch_size = position.shape[0]

        # Ensure control inputs are on the correct device
        omega_desired = omega_desired.to(self.device)
        thrust = thrust.to(self.device)

        # Apply rotor control noise if specified
        if self.rotor_noise_std is not None:
            thrust_noise = torch.randn_like(thrust) * self.rotor_noise_std
            thrust = thrust + thrust_noise

        # Apply rotor control noise if specified
        if self.br_noise_std is not None:
            omega_noise = torch.randn_like(omega_desired) * self.br_noise_std
            omega_desired = omega_desired + omega_noise
            
        # Ensure thrust is within [0, 1]
        thrust = torch.clamp(thrust, 0.0, 1.0)

        # Compute angular velocity error
        omega_error = omega_desired - angular_velocity

        # Compute control torque using PD controller
        Tau = self.Kp * omega_error - self.Kd * angular_velocity
        
        # Compute angular acceleration
        # Reshape I and I_inv for batch processing
        I = self.I.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, 3, 3)
        I_inv = self.I_inv.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, 3, 3)

        omega_cross = torch.cross(angular_velocity.clone(), torch.bmm(I, angular_velocity.clone().unsqueeze(2)).squeeze(2))
        omega_dot = torch.bmm(I_inv, (Tau - omega_cross).unsqueeze(2)).squeeze(2)

        # print('***********************************************')
        # print(f'omega_desired: {omega_desired}')
        # print(f'angular_velocity: {angular_velocity}')
        # print(f'omega_error: {omega_error}')
        # print(f'Tau: {Tau}')
        # print(f'omega_dot:{omega_dot}')


        # Update angular velocity
        next_angular_velocity = angular_velocity + omega_dot * self.dt

        # Update orientation quaternion
        next_orientation = self.integrate_quaternion(orientation, next_angular_velocity, self.dt)

        # Compute rotation matrix from quaternion
        R = self.quaternion_to_rotation_matrix(next_orientation)  # Shape: (batch_size, 3, 3)

        # Compute actual thrust force
        actual_thrust = thrust * self.max_thrust  # thrust is normalized between 0 and 1

        # Compute thrust force in body frame
        F_body = torch.zeros((batch_size, 3), device=self.device)
        F_body[:, 2] = actual_thrust.squeeze()  # Thrust along z-axis

        # Transform thrust force to world frame
        F_world = torch.bmm(R, F_body.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, 3)

        # Compute linear acceleration
        linear_acceleration = F_world / self.m + self.g

        # Update velocity and position
        next_velocity = velocity + linear_acceleration * self.dt
        next_position = position + next_velocity * self.dt

        return next_position, next_velocity, next_orientation, next_angular_velocity, linear_acceleration, omega_dot

    def run_simulation(self, position, velocity, orientation, angular_velocity, control_input):
        """
        Simulate multiple time steps given the current state and control inputs.

        Args:
            position (torch.Tensor): Current position (batch_size, 3)
            velocity (torch.Tensor): Current velocity (batch_size, 3)
            orientation (torch.Tensor): Current orientation quaternion (batch_size, 4)
            angular_velocity (torch.Tensor): Current angular velocity (batch_size, 3)
            control_input (tuple): A tuple of (omega_desired, thrust), both are tensors of shape (batch_size, ...)

        Returns:
            final_position (torch.Tensor): Final position after simulation (batch_size, 3)
            final_velocity (torch.Tensor): Final velocity after simulation (batch_size, 3)
            final_orientation (torch.Tensor): Final orientation quaternion after simulation (batch_size, 4)
            final_linear_acceleration (torch.Tensor): Final linear acceleration (batch_size, 3)
        """
        omega_desired, thrust = control_input
        # The control inputs are constant over the simulation

        # Simulation loop
        for step in range(self.n_steps):
            # Update the quadrotor's state
            position, velocity, orientation, angular_velocity, linear_acceleration, angular_acceleration = self.update(
                position, velocity, orientation, angular_velocity, omega_desired, thrust)

        # Only output the final acceleration
        final_position = position
        final_linear_velocity = velocity
        final_angular_velocity = angular_velocity
        final_orientation = orientation
        final_linear_acceleration = linear_acceleration  
        final_angular_acceleration = angular_acceleration # Shape: (batch_size, 3)

        return final_position, final_linear_velocity, final_angular_velocity, final_orientation, final_linear_acceleration, final_angular_acceleration

if __name__ == '__main__':
    # Initialize the simulator
    simulator = QuadrotorSimulator(
        mass=1.,
        inertia=torch.diag(torch.tensor([0.005, 0.005, 0.01])),
        link_length=0.15,
        Kp=torch.tensor([1.0, 1.0, 2.0]),
        Kd=torch.tensor([0.03, 0.03, 0.04]),
        freq=200.0,
        max_thrust=30.0,  # Maximum thrust in Newtons
        total_time=0.05,  # Total simulation time in seconds
        rotor_noise_std=0.01,  # Standard deviation for rotor control noise
        br_noise_std=0.01  # Standard deviation for body rate control noise
    )

    # Batch size
    batch_size = 1

    # Initial state (batch_size, dim)
    position = torch.zeros((batch_size, 3), device=simulator.device)
    velocity = torch.zeros((batch_size, 3), device=simulator.device)
    orientation = torch.zeros((batch_size, 4), device=simulator.device)
    orientation[:, 0] = 1.0  # Quaternion [w, x, y, z]
    position[:,2] = 1.2
    angular_velocity = torch.zeros((batch_size, 3), device=simulator.device)

    # Control inputs (batch_size, dim)
    omega_desired = torch.zeros((batch_size, 3), device=simulator.device)
    omega_desired[:,0] = 0.75
    omega_desired[:,1] = 0.25
    omega_desired[:,2] = 0.85
    thrust = torch.full((batch_size, 1), 0.33, device=simulator.device)  # Normalized thrust

    control_input = (omega_desired, thrust)

    # Run simulation
    start_time = time.time()
    final_position, final_velocity, final_angular_velocity, final_orientation, final_lin_acc, final_ang_acc = simulator.run_simulation(
        position, velocity, orientation, angular_velocity, control_input
    )
    elapsed_time = time.time() - start_time
    # print(elapsed_time)

    # Outputs
    
    print("Device:", simulator.device)
    print("Elapsed time:", elapsed_time)
    print("Final Position:", final_position)
    print("Final Linear Velocity:", final_velocity)
    print("Final Angular Velocity:", final_angular_velocity)
    print("Final Orientation:", final_orientation)
    print("Final Linear Acceleration:", final_lin_acc)
    print("Final Angular Acceleration:", final_ang_acc)
import torch
import time
import pdb

class QuadrotorSimulator:
    def __init__(self, mass=1.0, inertia=None, link_length=0.1, Kp=None, Kd=None, 
                 freq=100.0, max_thrust=15.0, total_time=1.0, rotor_noise_std=0.01, br_noise_std=0.01,
                 drag_coeff=0.5, cross_area=0.1, air_density=1.225, motor_time_constant=0.02, torque_constant=0.05):
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
            motor_time_constant (float): Motor time constant.
            torque_constant (float): Torque constant for torque-force coupling.
        """
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Quadrotor dynamics properties
        self.m = mass.to(self.device)  # Shape: (batch_size,)
        if inertia is None:
            # Default to a simple diagonal inertia matrix
            self.I = torch.diag_embed(torch.tensor([0.01, 0.01, 0.02], device=self.device)).unsqueeze(0).repeat(mass.shape[0], 1, 1)
        else:
            self.I = inertia.to(self.device)  # Shape: (batch_size, 3, 3)
        self.I_inv = torch.inverse(self.I)  # Shape: (batch_size, 3, 3)
        self.l = torch.tensor(link_length, device=self.device)  # Link length
        self.max_thrust = max_thrust.to(self.device)  # Shape: (batch_size,)

        # PD gains for angular rate control
        if Kp is None:
            self.Kp = torch.tensor([1.0, 1.0, 1.0], device=self.device).unsqueeze(0).repeat(mass.shape[0], 1)
        else:
            self.Kp = Kp.to(self.device)  # Shape: (batch_size, 3)
        if Kd is None:
            self.Kd = torch.tensor([0.1, 0.1, 0.1], device=self.device).unsqueeze(0).repeat(mass.shape[0], 1)
        else:
            self.Kd = Kd.to(self.device)  # Shape: (batch_size, 3)

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

        # Motor dynamics
        self.motor_time_constant = motor_time_constant
        self.torque_constant = torque_constant

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

        # Apply motor dynamics (update motor speeds based on desired thrust)
        # desired_motor_speeds = torch.sqrt(thrust / self.max_thrust.unsqueeze(1))  # Convert thrust to motor speeds
        # motor_speeds = motor_speeds + (desired_motor_speeds - motor_speeds) * (self.dt / self.motor_time_constant)

        # Apply rotor control noise
        if self.rotor_noise_std is not None:
            thrust_noise = torch.randn_like(thrust) * self.rotor_noise_std

        # Compute actual thrust from motor speeds
        # actual_thrust = motor_speeds**2 * self.max_thrust.unsqueeze(1)
        # print(actual_thrust)
        actual_thrust = (thrust + thrust_noise) * self.max_thrust.clone()

        # Apply body rate noise
        if self.br_noise_std is not None:
            omega_noise = torch.randn_like(omega_desired_body) * self.br_noise_std
            omega_desired_body = omega_desired_body + omega_noise

        # Compute angular velocity error
        omega_error = omega_desired_body - angular_velocity
        est_alpha = (angular_velocity - self.prev_omega) / self.dt
        est_alpha = est_alpha.detach()

        # print(angular_velocity - self.prev_omega)

        # Compute control torque using PD controller(suppose desired alpha = 0)
        # TODO： check this part
        Tau = self.Kp * omega_error - self.Kd * est_alpha

        # Compute angular acceleration
        omega_cross = torch.cross(angular_velocity.clone(), torch.bmm(self.I.clone().detach(), angular_velocity.clone().unsqueeze(2)).squeeze(2))
        omega_dot = torch.bmm(self.I_inv.clone().detach(), (Tau - omega_cross).unsqueeze(2)).squeeze(2)

        # Update angular velocity
        next_angular_velocity = angular_velocity + omega_dot * self.dt

        # Update orientation quaternion
        next_orientation = self.integrate_quaternion(orientation, next_angular_velocity, self.dt)

        # Compute rotation matrix
        R = self.quaternion_to_rotation_matrix(next_orientation)

        # Compute thrust force in body frame
        F_body = torch.zeros((batch_size, 3), device=self.device)
        F_body[:, 2] = actual_thrust.clone()

        # Transform thrust force to world frame
        F_world = torch.bmm(R, F_body.unsqueeze(2)).squeeze(2)

        # Compute drag force
        drag_force = -0.5 * self.drag_coeff * self.cross_area * self.air_density * velocity.clone() * torch.norm(velocity.clone(), dim=1, keepdim=True)

        # Compute linear acceleration
        linear_acceleration = (F_world + drag_force.clone()) / self.m.clone().unsqueeze(1) + self.g

        # Update velocity and position
        next_velocity = velocity + linear_acceleration * self.dt
        next_position = position + next_velocity * self.dt

        # record omega
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

        # Initialize motor speeds (batch_size, 1)
        motor_speeds = torch.zeros_like(thrust, device=self.device)

        # Simulation loop
        for step in range(self.n_steps):
            position, velocity, orientation, angular_velocity, linear_acceleration, angular_acceleration, motor_speeds = self.update(
                position, velocity, orientation, angular_velocity, omega_desired_body, thrust, motor_speeds
            )

        # Return final states
        return position, velocity, angular_velocity, orientation, linear_acceleration, angular_acceleration
    
# Example usage
if __name__ == '__main__':
    batch_size = 2

    # Initialize the simulator
    # simulator = QuadrotorSimulator(
    #     mass=torch.tensor([1.0, 1.2]),  # Different masses for each drone
    #     inertia=torch.diag_embed(torch.tensor([[0.02, 0.02, 0.04], [0.06, 0.06, 0.12]])),  # Different inertias
    #     link_length=0.15,
    #     Kp=torch.tensor([[2.9, 2.45, 3.5], [2.9, 2.45, 3.5]]),  # Different Kp gains
    #     Kd=torch.tensor([[0.003, 0.0025, 0.0025], [0.003, 0.0025, 0.0025]]),  # Different Kd gains
    #     freq=200.0,
    #     max_thrust=torch.tensor([25.0, 35.0]),  # Different max thrusts
    #     total_time=0.05,
    #     rotor_noise_std=0.01,
    #     br_noise_std=0.01
    # )

    # simulator = QuadrotorSimulator(
    #     mass=torch.tensor([1.15, 1.2]),  # Different masses for each drone
    #     inertia=torch.diag_embed(torch.tensor([[0.008, 0.012, 0.025], [0.06, 0.06, 0.12]])),  # Different inertias
    #     link_length=0.15,
    #     Kp=torch.tensor([[0.8, 1.2, 2.5], [2.9, 2.45, 3.5]]),  # Different Kp gains
    #     Kd=torch.tensor([[0.001, 0.001, 0.002], [0.003, 0.0025, 0.0025]]),  # Different Kd gains
    #     freq=200.0,
    #     max_thrust=torch.tensor([25.0, 35.0]),  # Different max thrusts
    #     total_time=0.05,
    #     rotor_noise_std=0.01,
    #     br_noise_std=0.01,
    # )

    simulator = QuadrotorSimulator(
        mass=torch.tensor([1.15, 1.2]),  # Different masses for each drone
        inertia=torch.diag_embed(torch.tensor([[0.01, 0.012, 0.025], [0.06, 0.06, 0.12]])),  # Different inertias
        link_length=0.15,
        Kp=torch.tensor([[1.0, 1.2, 2.5], [2.9, 2.45, 3.5]]),  # Different Kp gains
        Kd=torch.tensor([[0.001, 0.001, 0.002], [0.003, 0.0025, 0.0025]]),  # Different Kd gains
        freq=200.0,
        max_thrust=torch.tensor([25.0, 35.0]),  # Different max thrusts
        total_time=0.05,
        rotor_noise_std=0.01,
        br_noise_std=0.01,
    )

    # Initial state
    position = torch.zeros((batch_size, 3), device=simulator.device)
    velocity = torch.zeros((batch_size, 3), device=simulator.device)
    orientation = torch.zeros((batch_size, 4), device=simulator.device)
    orientation[:, 0] = 1.0  # Quaternion [w, x, y, z]
    position[:, 2] = 1.2  # Initial height
    angular_velocity = torch.zeros((batch_size, 3), device=simulator.device)

    # Control inputs
    omega_desired_body = torch.zeros((batch_size, 3), device=simulator.device)
    omega_desired_body[:, 0] = 0.35
    omega_desired_body[:, 1] = -0.5
    omega_desired_body[:, 2] = 0.15
    thrust = torch.full((batch_size,), 0.45, device=simulator.device)  # Normalized thrust

    # Run simulation
    start_time = time.time()
    final_position, final_velocity, final_angular_velocity, final_orientation, final_lin_acc, final_ang_acc = simulator.run_simulation(
        position, velocity, orientation, angular_velocity, (omega_desired_body, thrust)
    )
    elapsed_time = time.time() - start_time

    # Outputs
    print("Device:\n", simulator.device)
    print("Elapsed time:\n", elapsed_time)
    print("Final Position:\n", final_position)
    print("Final Linear Velocity:\n", final_velocity)
    print("Final Angular Velocity:\n", final_angular_velocity)
    print("Final Orientation:\n", final_orientation)
    print("Final Linear Acceleration:\n", final_lin_acc)
    print("Final Angular Acceleration:\n", final_ang_acc)
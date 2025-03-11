import time
from robot import Robot
from mpc import RobotMPC

# Initialize robot and MPC controller
robot = Robot(X=[0, 0, 0, 0, 0, 0])
mpc_controller = RobotMPC(robot)

# Example reference state for tracking
reference_state = [1, 0, 0, 0, 0, 0]  # Desired state (x, dx, theta, dtheta, psi, dpsi)

# Simulate the robot's motion using the MPC controller
for i in range(1000):
    # Solve for the optimal control inputs
    optimal_control = mpc_controller.solve(reference_state)

    # Apply the optimal control to the robot
    left_torque, right_torque = optimal_control
    robot.updateConfiguration(left_torque, right_torque, 0.02)

    # Print the robot's position at each step
    print(f"Time: {i * 0.02}, Position: {robot.getPosition()}")
    time.sleep(0.02)

import time
from robot import Robot
from mpc import RobotMPC


robot = Robot(X=[0, 0, 0, 0, 0, 0])
mpc_controller = RobotMPC(robot)

reference_state = [1, 0, 0, 0, 0, 0]  # Desired state (x, dx, theta, dtheta, psi, dpsi)

for i in range(1000):
    optimal_control = mpc_controller.solve(reference_state)
    left_torque, right_torque = optimal_control
    robot.updateConfiguration(left_torque, right_torque, 0.02)
    print(f"Time: {i * 0.02}, Position: {robot.getPosition()}")
    time.sleep(0.02)

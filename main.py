import time
from robot import Robot
from mpc import RobotMPC


robot = Robot(X=[0, 0, 0, 0, 0, 0])
mpc_controller = RobotMPC(robot)

reference_state = [10, 0, 0, 0, 0, 0] 

for i in range(1000):
    optimal_control = mpc_controller.solve(reference_state)
    left_torque, right_torque = optimal_control
    robot.updateConfiguration(0.1,0.1, 0.02)
    print(f"Time1: {i * 0.02}, Position: {robot.getPosition()}, Control: {optimal_control}")
    time.sleep(0.02)

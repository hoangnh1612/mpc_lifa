import time
from robot import Robot
from mpc import RobotMPC
import numpy as np
from params import * 
N = 30
robot = Robot(X=[0, 0, 0, 0, 0, 0])
mpc_controller = RobotMPC(robot)

reference_state = [0.5, 0, 0, 0, 0, 0] 

for i in range(1000):
    optimal_control = mpc_controller.solve(reference_state)
    predicted_next_state = mpc_controller.opti.value(mpc_controller.opt_states[1, :])
    actual_next_state = robot.getState()
    left_torque, right_torque = optimal_control
    robot.updateConfiguration(left_torque,right_torque, T)
    print(f"Time1: {i * T}, Position: {robot.getPosition()}, Control: {optimal_control}")
    print(f"Predicted next state (MPC): {predicted_next_state}")
    print(f"Actual next state (Robot): {actual_next_state}")
    print(f"Difference: {np.abs(predicted_next_state - actual_next_state)}")
    print("-" * 50)
    time.sleep(0.02)

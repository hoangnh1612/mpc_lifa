import casadi as ca
import numpy as np
import math
import time

T = 0.02
N = 50

class RobotMPC:
    def __init__(self, robot, T=T, N=N, Q=np.diag([50.0, 0.5, 10.0, 0.5, 1.5, 0.5]), R=np.diag([1.0, 1.0])):
        self.robot = robot
        self.T = T
        self.N = N
        self.Q = Q
        self.R = R
        
        self.next_states = np.tile(self.robot.X, (self.N + 1, 1))
        self.u0 = np.ones((self.N, 2)) * 0.5
        self.setup_controller()

    def dynamics(self, x_, dx_, theta_, dtheta_, psi_, dpsi_, u_):
        A = self.robot.A
        B = self.robot.B
        state_term = ca.mtimes(A, ca.vertcat(x_, dx_, theta_, dtheta_, psi_, dpsi_))
        control_term = ca.mtimes(B, u_.T)
        f = state_term + control_term
        return f
    
    def setup_controller(self):
        self.opti = ca.Opti()
        self.opt_controls = self.opti.variable(self.N, 2)
        self.current_state = self.opti.parameter(6, 1)
        self.opt_states = self.opti.variable(self.N + 1, 6)
        self.reference_state = self.opti.parameter(6, 1)
        
        self.opti.subject_to(self.opt_states[0, :] == self.current_state.T)
        for i in range(self.N):
            x_ = self.opt_states[i, 0]
            dx_ = self.opt_states[i, 1]
            theta_ = self.opt_states[i, 2]
            dtheta_ = self.opt_states[i, 3]
            psi_ = self.opt_states[i, 4]
            dpsi_ = self.opt_states[i, 5]
            u_ = self.opt_controls[i, :]
            next_state = self.opt_states[i, :] + self.dynamics(x_, dx_, theta_, dtheta_, psi_, dpsi_, u_).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == next_state)

        obj = 0
        for i in range(self.N):
            state_error = self.opt_states[i, :] - self.reference_state.T
            control_effort = self.opt_controls[i, :]
            state_cost = ca.mtimes([state_error, self.Q, state_error.T])
            control_cost = ca.mtimes([control_effort, self.R, control_effort.T])
            obj += state_cost + control_cost
        terminal_error = self.opt_states[self.N, :] - self.reference_state.T
        obj += ca.mtimes([terminal_error, self.Q * 10, terminal_error.T])
        
        self.opti.minimize(obj)
        
        self.opti.subject_to(self.opti.bounded(-np.pi/3, self.opt_states[:, 2], np.pi/3))
        self.opti.subject_to(self.opti.bounded(-1.0, self.opt_controls[:, 0], 1.0))
        self.opti.subject_to(self.opti.bounded(-1.0, self.opt_controls[:, 1], 1.0))
        self.opti.subject_to(self.opti.bounded(-1.0, self.opt_states[:, 1], 1.0))
        self.opti.subject_to(self.opti.bounded(-2.5, self.opt_states[:, 3], 2.5))
        
        opts_setting = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-6,
            'ipopt.acceptable_obj_change_tol': 1e-3
        }
        self.opti.solver('ipopt', opts_setting)

    def solve(self, reference_state):
        # Ensure self.robot.X is a 6x1 column vector
        self.opti.set_value(self.current_state, self.robot.X.reshape(6, 1))
        self.opti.set_value(self.reference_state, np.array(reference_state).reshape(6, 1))
        
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)
        
        try:
            sol = self.opti.solve()
            u_res = sol.value(self.opt_controls)
            x_res = sol.value(self.opt_states)
            
            self.u0 = np.zeros((self.N, 2))
            self.u0[:-1, :] = u_res[1:, :]
            self.next_states = x_res
            
            return u_res[0]
        except Exception as e:
            print(f"Optimization failed: {e}")
            if 'Infeasible_Problem_Detected' in str(e):
                print("Debugging infeasibility:")
                print(f"Current state: {self.robot.X}")
                print(f"Reference state: {reference_state}")
                return np.array([0.5, 0.5])
            return np.array([0, 0])

class Robot:
    def __init__(self, X=[0, 0, 0, 0, 0, 0]):
        self.X = np.array(X, dtype=float).reshape(6,)  # Ensure 6-element vector
        self.A = np.array([[0, 1, 0, 0, 0, 0],
                           [0, -0.112, -2.45, 0.0135, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0.37, 23.245, -0.045, 0, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, -0.0337]])
        self.B = np.array([[0, 0],
                           [1.1869, 1.1869],
                           [0, 0],
                           [-3.932, -3.932],
                           [0, 0],
                           [2.5, -2.5]])
        self.max_theta = np.pi / 3
        self.min_theta = -self.max_theta
        self.max_dx = 1.0
        self.min_dx = -self.max_dx
        self.max_dtheta = 2.5
        self.min_dtheta = -self.max_dtheta
        self.max_dpsi = 0.8
        self.min_dpsi = -self.max_dpsi
        self.max_tau_right = 1.0
        self.min_tau_right = -self.max_tau_right
        self.max_tau_left = 1.0
        self.min_tau_left = -self.max_tau_left
        self.I = np.identity(6)

    def correctControl(self, rightTorque, leftTorque):
        rightTorque = np.clip(rightTorque, self.min_tau_right, self.max_tau_right)
        leftTorque = np.clip(leftTorque, self.min_tau_left, self.max_tau_left)
        return rightTorque, leftTorque

    def updateConfiguration(self, rightTorque, leftTorque, dt):
        rightTorque, leftTorque = self.correctControl(rightTorque, leftTorque)
        dX = ca.mtimes(self.A, self.X) + ca.mtimes(self.B, np.array([rightTorque, leftTorque]).T)
        self.X = self.X + np.array(dX, dtype=float).reshape(6,) * dt  # Ensure 6-element vector

    def getPosition(self):
        return self.X[0], self.X[2]

    def getState(self):
        return self.X

robot = Robot(X=[0, 0, 0, 0, 0, 0])
mpc_controller = RobotMPC(robot)

reference_state = [10, 0, 0, 0, 0, 0]

for i in range(100):
    optimal_control = mpc_controller.solve(reference_state)
    predicted_next_state = mpc_controller.next_states[1, :]
    actual_next_state = robot.getState()
    left_torque, right_torque = optimal_control
    robot.updateConfiguration(left_torque, right_torque, T)
    print(f"Time: {i * T:.3f}, Position: {robot.getPosition()}, Control: {optimal_control}")
    # print(f"Predicted next state (MPC): {predicted_next_state}")
    # print(f"Actual next state (Robot): {actual_next_state}")
    # print(f"Difference: {np.abs(predicted_next_state - actual_next_state)}")
    # print("-" * 50)
    time.sleep(0.02)
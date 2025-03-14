import casadi as ca
import numpy as np
import math
class RobotMPC:
    def __init__(self, robot, T=0.02, N=30, Q=np.diag([4.0, 0.0, 50.0, 0.0, 1.5, 0.0]), R=np.diag([1.0, 1.0])):
        self.robot = robot
        self.T = T  # Time step
        self.N = N  # Horizon length
        self.Q = Q
        self.R = R
        
        self.next_states = np.zeros((self.N + 1, 6))  
        self.u0 = np.zeros((self.N, 2))  # Initial control guess
        self.setup_controller()
    def dynamics(self, x_, dx_, theta_, dtheta_, psi_, dpsi_, u_):
        A = self.robot.A
        B = self.robot.B
        I = np.eye(6)
        
        state_term = ca.mtimes((I + A * self.T), ca.vertcat(x_, dx_, theta_, dtheta_, psi_, dpsi_))
        control_term = ca.mtimes(B * self.T, u_.T)
        f = state_term + control_term
        return f
    
    def setup_controller(self):
        self.opti = ca.Opti()
        self.opt_controls = self.opti.variable(self.N, 2)  # (N x 2) control inputs (torques)
        self.state0 = self.opti.parameter(6, 1)  # Initial state
        self.opt_states = self.opti.variable(self.N , 6)  # (N+1 x 6) states
        self.reference_state = self.opti.parameter(6, 1)  # Column vector parameter for reference state
        for i in range(self.N-1):
            # Explicitly unpack state variables
            x_ = self.opt_states[i, 0]
            dx_ = self.opt_states[i, 1]
            theta_ = self.opt_states[i, 2]
            dtheta_ = self.opt_states[i, 3]
            psi_ = self.opt_states[i, 4]
            dpsi_ = self.opt_states[i, 5]
            u_ = self.opt_controls[i, :]
            next_state = self.opt_states[i, :] + self.dynamics(x_, dx_, theta_, dtheta_, psi_, dpsi_, u_).T * self.T
            # print(i,"OP: ",self.opt_states[i + 1, :].shape,"NS: ", next_state.shape)
            self.opti.subject_to(self.opt_states[i + 1, :] == next_state)
        obj = 0
        for i in range(self.N):
            state_error = self.opt_states[i, :] - self.reference_state.T
            control_effort = self.opt_controls[i, :]
            state_cost = ca.mtimes([state_error, self.Q, state_error.T])
            control_cost = ca.mtimes([control_effort, self.R, control_effort.T])
            obj += state_cost + control_cost
        
        self.opti.minimize(obj)
        
        self.opti.subject_to(self.opti.bounded(-np.pi / 5, self.opt_states[:, 2], np.pi / 5))  # theta constraints
        self.opti.subject_to(self.opti.bounded(-1.5, self.opt_controls[:, 0], 1.5))  # right torque
        self.opti.subject_to(self.opti.bounded(-1.5, self.opt_controls[:, 1], 1.5))  # left torque
        self.opti.subject_to(self.opti.bounded(-0.5, self.opt_states[:, 1], 0.5))  # dx constraints
        self.opti.subject_to(self.opti.bounded(-1, self.opt_states[:, 3], 1)) 
        
        opts_setting = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-3
        }
        self.opti.solver('ipopt', opts_setting)
        self.opti.subject_to(self.opt_states[0, :].T == self.robot.X)
    def solve(self, reference_state):
        # self.opti.subject_to(self.opt_states[0, :] == np.array([self.robot.X]))  
        self.opti.set_value(self.state0, self.robot.X)
        self.opti.set_value(self.reference_state, np.array(reference_state).reshape(6, 1))
        # self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)
        try:
            sol = self.opti.solve()
            u_res = sol.value(self.opt_controls)
            x_res = sol.value(self.opt_states)
            self.u0 = np.zeros((self.N, 2))
            if self.N > 1:
                self.u0[0:self.N-1, :] = u_res[1:self.N, :]
            self.next_states = np.zeros((self.N+1, 6))
            self.next_states[0:self.N, :] = x_res[1:self.N+1, :]
            return u_res[0]
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return np.array([0, 0])

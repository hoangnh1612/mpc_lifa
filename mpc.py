import casadi as ca
import numpy as np
import math
from params import * 
class RobotMPC:
    def __init__(self, robot, T=T, N=30, Q=np.diag([4.0, 0.5, 50.0, 0.5, 1.5, 0.5]), R=np.diag([1.0, 1.0])):
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
        self.current_state = self.opti.parameter(6, 1)  # Initial state
        self.opt_states = self.opti.variable(self.N + 1, 6)  # (N+1 x 6) states
        self.reference_state = self.opti.parameter(6, 1)  # Column vector parameter for reference state
        self.opti.subject_to(self.opt_states[0, :] == self.current_state.T)
        for i in range(self.N):
            x_ = self.opt_states[i, 0]
            dx_ = self.opt_states[i, 1]
            theta_ = self.opt_states[i, 2]
            dtheta_ = self.opt_states[i, 3]
            psi_ = self.opt_states[i, 4]
            dpsi_ = self.opt_states[i, 5]
            u_ = self.opt_controls[i, :]
            next_state = self.dynamics(x_, dx_, theta_, dtheta_, psi_, dpsi_, u_).T
            self.opti.subject_to(self.opt_states[i + 1, :] == next_state)
        obj = 0
        for i in range(self.N):
            state_error = self.opt_states[i, :] - self.reference_state.T
            control_effort = self.opt_controls[i, :]
            state_cost = ca.mtimes([state_error, self.Q, state_error.T])
            control_cost = ca.mtimes([control_effort, self.R, control_effort.T])
            obj += state_cost + control_cost
            # obj += state_cost
        
        self.opti.minimize(obj)
        
        self.opti.subject_to(self.opti.bounded(-np.pi / 4, self.opt_states[:, 2], np.pi / 4))  # theta constraints
        self.opti.subject_to(self.opti.bounded(-1.5, self.opt_controls[:, 0], 1.5))  # right torque
        self.opti.subject_to(self.opti.bounded(-1.5, self.opt_controls[:, 1], 1.5))  # left torque
        self.opti.subject_to(self.opti.bounded(-0.8, self.opt_states[:, 1], 0.8))  # dx constraints
        self.opti.subject_to(self.opti.bounded(-2, self.opt_states[:, 3], 2)) 
        
        opts_setting = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-3
        }
        self.opti.solver('ipopt', opts_setting)
    def solve(self, reference_state):
            self.opti.set_value(self.current_state, self.robot.X)
            self.opti.set_value(self.reference_state, np.array(reference_state).reshape(6, 1))
            
            # Warm start with shifted previous solution
            self.opti.set_initial(self.opt_states, self.next_states)
            self.opti.set_initial(self.opt_controls, self.u0)
            
            try:
                sol = self.opti.solve()
                u_res = sol.value(self.opt_controls)
                x_res = sol.value(self.opt_states)
                
                # Shift solutions for warm start
                self.u0 = np.zeros((self.N, 2))
                self.u0[:-1, :] = u_res[1:, :]
                self.next_states = x_res
                
                return u_res[0]
            except Exception as e:
                print(f"Optimization failed: {e}")
                # Try to debug infeasibility
                if 'Infeasible_Problem_Detected' in str(e):
                    print("Debugging infeasibility:")
                    print(f"Current state: {self.robot.X}")
                    print(f"Reference state: {reference_state}")
                    try:
                        sol = self.opti.solve()
                        return sol.value(self.opt_controls)[0]
                    except:
                        return np.array([0, 0])
                return np.array([0, 0])

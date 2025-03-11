# mpc.py
import casadi as ca
import numpy as np
import math

class RobotMPC:
    def __init__(self, robot, T=0.02, N=30, Q=np.diag([40.0, 1.0, 0.5, 0.1, 0.1, 0.1]), R=np.diag([1.0, 1.0])):
        self.robot = robot
        self.T = T  # Time step
        self.N = N  # Horizon length
        
        # Cost function weights
        self.Q = Q
        self.R = R
        
        # Initialize state and control guesses
        self.next_states = np.zeros((self.N + 1, 6))  # Including 0th state
        self.u0 = np.zeros((self.N, 2))  # Initial control guess
        
        self.setupController()
        
    def setupController(self):
        # Create CasADi optimization problem
        self.opti = ca.Opti()
        
        # Define the decision variables (controls and states)
        self.opt_controls = self.opti.variable(self.N, 2)  # (N x 2) control inputs (torques)
        self.opt_states = self.opti.variable(self.N + 1, 6)  # (N+1 x 6) states
        
        # Define the reference state as a parameter within the Opti context
        self.reference_state = self.opti.parameter(6, 1)  # Column vector parameter for reference state
        
        # Set up dynamics for each step in the horizon
        for i in range(self.N):
            # Explicitly unpack state variables
            x_ = self.opt_states[i, 0]
            dx_ = self.opt_states[i, 1]
            theta_ = self.opt_states[i, 2]
            dtheta_ = self.opt_states[i, 3]
            psi_ = self.opt_states[i, 4]
            dpsi_ = self.opt_states[i, 5]
            
            # Unpack control input
            u_ = self.opt_controls[i, :]
            
            # Define the system dynamics
            def dynamics(x_, dx_, theta_, dtheta_, psi_, dpsi_, u_):
                A = self.robot.A
                B = self.robot.B
                I = np.eye(6)
                
                # Fixing the matrix multiplication
                state_term = ca.mtimes((I + A * self.T), ca.vertcat(x_, dx_, theta_, dtheta_, psi_, dpsi_))
                control_term = ca.mtimes(B * self.T, u_)  # u_ is already a vector
                f = state_term + control_term
                return f
            
            # Predict the next state based on the dynamics
            next_state = self.opt_states[i, :] + dynamics(x_, dx_, theta_, dtheta_, psi_, dpsi_, u_).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == next_state)
        
        # Define the cost function: minimize state error and control effort
        obj = 0
        for i in range(self.N):
            # State error (difference between current state and reference state)
            state_error = self.opt_states[i, :] - self.reference_state.T
            control_effort = self.opt_controls[i, :]
            
            # Quadratic costs
            state_cost = ca.mtimes([state_error, self.Q, state_error.T])
            control_cost = ca.mtimes([control_effort, self.R, control_effort.T])
            
            obj += state_cost + control_cost
        
        self.opti.minimize(obj)
        
        # Constraints on states
        self.opti.subject_to(self.opti.bounded(-np.pi / 4, self.opt_states[:, 2], np.pi / 4))  # theta constraints
        self.opti.subject_to(self.opti.bounded(-2, self.opt_states[:, 3], 2))  # dtheta constraints
        
        # Torque constraints
        self.opti.subject_to(self.opti.bounded(-1.5, self.opt_controls[:, 0], 1.5))  # right torque
        self.opti.subject_to(self.opti.bounded(-1.5, self.opt_controls[:, 1], 1.5))  # left torque
        
        # Solver settings
        opts_setting = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        
        self.opti.solver('ipopt', opts_setting)
    
    def solve(self, reference_state):
        # Set initial state constraint (current robot state)
        self.opti.subject_to(self.opt_states[0, :] == self.robot.X)
        
        # Set the reference state parameter
        self.opti.set_value(self.reference_state, np.array(reference_state).reshape(6, 1))
        
        # Initial guess for the optimization variables
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)
        
        try:
            # Solve the optimization problem
            sol = self.opti.solve()
            
            # Extract the optimal control inputs and states
            u_res = sol.value(self.opt_controls)
            x_res = sol.value(self.opt_states)
            
            # Update the control and state history for the next iteration
            self.u0 = np.zeros((self.N, 2))
            if self.N > 1:
                self.u0[0:self.N-1, :] = u_res[1:self.N, :]
            
            self.next_states = np.zeros((self.N+1, 6))
            self.next_states[0:self.N, :] = x_res[1:self.N+1, :]
            
            return u_res[0]
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return np.array([0, 0])

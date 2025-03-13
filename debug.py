import casadi as ca
import numpy as np
import time
    
class SimpleRobot:
    def __init__(self):
        self.x = 0
        self.dx = 0
        self.X = np.array([self.x, self.dx])
    def updateConfiguration(self, v, dt):
        self.x += v * dt
        self.dx = v
        self.X = np.array([self.x, self.dx])
    def getPosition(self):
        return self.x, self.dx

class MPCSimpleRobot:
    def __init__(self, robot, T=0.02, N=30, Q=np.diag([4.0, 1.0]), R=np.diag([1.0])):
        self.robot = robot
        self.T = T  # Time step
        self.N = N  # Horizon length
        self.Q = Q
        self.R = R
        
        self.next_states = np.zeros((self.N + 1, 2))  # Including 0th state
        self.u0 = np.zeros((self.N,1))  # Initial control guess
        
    def dynamics(self, u_):
        return u_
    def solve(self, reference_state):
        self.opti = ca.Opti()
        self.opt_controls = self.opti.variable(self.N, 1)  
        self.opt_states = self.opti.variable(self.N + 1, 2) 
        self.reference_state = self.opti.parameter(2, 1) 
        for i in range(self.N):
            u_ = self.opt_controls[i]
            next_state = self.opt_states[i] + self.dynamics(u_).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == next_state)
        
        obj = 0
        # print("OptState: ", self.opt_states[0,:].shape,"RefState: ",self.reference_state.T.shape) 
        for i in range(self.N):
            state_error = self.opt_states[i,:] - self.reference_state.T
            print("SE: ",state_error.shape)
            control_effort = self.opt_controls[i]
            # print("CE: ",control_effort.shape)
            state_cost = ca.mtimes([state_error, self.Q, state_error.T])
            control_cost = ca.mtimes([control_effort, self.R, control_effort.T])
            
            obj += state_cost + control_cost
        
        self.opti.minimize(obj)
        
        self.opti.subject_to(self.opti.bounded(-1, self.opt_controls,1))  
        
        opts_setting = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        
        self.opti.solver('ipopt', opts_setting)
    
        self.opti.subject_to(self.opt_states[0, :].T == self.robot.X)
        
        self.opti.set_value(self.reference_state, np.array(reference_state).reshape(2, 1))
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)
        
        try:
            sol = self.opti.solve()
            u_res = sol.value(self.opt_controls)
            x_res = sol.value(self.opt_states)
            self.u0 = np.zeros((self.N, 1))
            if self.N > 1:
                self.u0[0:self.N-1, :] = u_res[1:self.N, :]
            self.next_states = np.zeros((self.N+1, 2))
            self.next_states[0:self.N, :] = x_res[1:self.N+1, :]
            return u_res[0]
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return np.array([0, 0])

robot = SimpleRobot()
mpc = MPCSimpleRobot(robot)
for i in range(1000):
    control = mpc.solve([1, 0])
    robot.updateConfiguration(control, 0.02)
    print(f"Time: {i * 0.02}, Position: {robot.getPosition()}")
    time.sleep(0.02)    


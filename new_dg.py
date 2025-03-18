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
    def get_x(self):
        return self.x
    
class Controller:
    def __init__(self, robot, Q=np.diag([20.0, 2.0]), R=np.diag([1.0])):
        self.robot = robot
        self.Q = Q
        self.R = R
        self.dt = 0.01
        self.prev_control = 0
    def solve(self, reference_state):
        self.opti = ca.Opti()
        self.state = self.opti.variable(2)
        self.control = self.opti.variable(1)
        self.reference_state = self.opti.parameter(2)
        obj = 0
        state_error = (self.state - self.reference_state).T
        control_effort = self.control
        state_cost = ca.mtimes([state_error,self.Q, state_error.T])
        control_cost = ca.mtimes([control_effort,self.R, control_effort.T])
        obj += state_cost + control_cost
        self.opti.minimize(obj)
        self.opti.subject_to(self.opti.bounded(-1, self.control, 1))
        opts_setting = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        self.opti.solver('ipopt', opts_setting)
        self.opti.set_initial(self.control, self.prev_control)
        self.opti.set_value(self.reference_state, np.array(reference_state).reshape(2, 1))
        self.opti.subject_to(self.state[0] == self.control*self.dt + np.array([self.robot.get_x()]))
        try:
            sol = self.opti.solve()
            self.prev_control = sol.value(self.control)
            return self.prev_control
        except Exception as e:
            print(f"Optimization failed: {e}")
            return np.array([0])
        
robot = SimpleRobot()
controller = Controller(robot)
reference_state = [-5.5, 0]
for i in range(1000):
    optimal_control = controller.solve(reference_state)
    robot.updateConfiguration(optimal_control, 0.02)
    print(f"Time: {i * 0.02}, Position: {robot.getPosition()}")
    time.sleep(0.02)
        
        
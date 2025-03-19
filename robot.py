import numpy as np
import time 
class Robot:
    def __init__(self, X = [0, 0, 0, 0, 0, 0]):
        # The configuration of TWIP 
        self.X = np.array(X)

        # Dynamic Metrics 
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

        # Constraints
        self.max_theta = np.pi / 4
        self.min_theta = -self.max_theta
        self.max_yaw = 2 * np.pi
        self.min_phi = -self.max_yaw

        self.max_dx = 0.8
        self.min_dx = -self.max_dx
        self.max_dtheta = 2
        self.min_dtheta = -self.max_dtheta
        self.max_dpsi = 0.8
        self.min_dpsi = -self.max_dpsi

        self.max_tau_right = 1.5
        self.min_tau_right = -self.max_tau_right
        self.max_tau_left = 1.5
        self.min_tau_left = -self.max_tau_left
        self.path = []
        self.I = np.identity(6)

    def correctControl(self, rightTorque, leftTorque):
        # Control limits for torques
        rightTorque = np.clip(rightTorque, self.min_tau_right, self.max_tau_right)
        leftTorque = np.clip(leftTorque, self.min_tau_left, self.max_tau_left)
        return rightTorque, leftTorque

    def updateConfiguration(self, rightTorque, leftTorque, dt):
        rightTorque, leftTorque = self.correctControl(rightTorque, leftTorque)
        dX = (self.I + self.A * dt) @ self.X + self.B @ np.array([rightTorque, leftTorque]) * dt
        self.X = self.X + dX * dt

    def getPosition(self):
        return self.X[0], self.X[2]

    def getState(self):
        return self.X

 
# robot = Robot(X=[0, 0, 0, 0, 0, 0])
# for i in range(1000):
#     robot.updateConfiguration(0.1, -0.1, 0.02)
#     print(f"Time: {i * 0.02}, Position: {robot.getPosition()}")
#     time.sleep(0.02)
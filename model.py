import numpy as np
import math
import time
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, X = [0, 0, 0, 0, 0, 0]):
        # The configuration of TWIP 
        self.X = X

        # Dynamic Metrics 
        self.A = np.array([[0, 1, 0, 0, 0, 0],\
                           [0, -0.112, -2.45, 0.0135, 0, 0],\
                           [0, 0, 0, 1, 0, 0],\
                           [0, 0.37, 23.245, -0.045, 0, 0],\
                           [0, 0, 0, 0, 0, 1],\
                           [0, 0, 0, 0, 0, -0.0337]])
        self.B = np.array([[0, 0],\
                           [1.1869, 1.1869],\
                           [0, 0],\
                           [-3.932, -3.932],\
                           [0, 0],\
                           [2.5, -2.5]])
        print(self.A[1])


        # The constraints of the TWIP
        self.max_theta = 3.14/4; self.min_theta = -self.max_theta
        self.max_yaw = 2*3.14; self.min_phi = -self.max_yaw

        self.max_dx = 0.8; self.min_dx = -self.max_dx
        self.max_dtheta = 2; self.min_dtheta = -self.max_dtheta
        self.max_dpsi = 0.8; self.min_dpsi = -self.max_dpsi

        self.max_tau_right = 1.5; self.min_tau_right = -self.max_tau_right
        self.max_tau_left = 1.5; self.min_tau_left = -self.max_tau_left
        self.path = []
        self.I = np.identity(6)

    def correctControl(self, rightTorque, leftTorque):
        rightTorque = min(max(rightTorque, self.min_tau_right), self.max_tau_right)
        leftTorque = min(max(leftTorque, self.min_tau_left), self.max_tau_left)

        return rightTorque, leftTorque
    

    def updateConfiguration(self, rightTorque, leftTorque, dt):
        # Prepare the current state
        x = self.X[0] # 
        dx = self.X[1]
        theta = self.X[2] # < pi/4
        dtheta = self.X[3] 

        psi = self.X[4] # rotation angle
        dpsi = self.X[5]
        self.X = np.array([x, dx, theta, dtheta, psi, dpsi])
        # The dynamic equations
        rightTorque, leftTorque = self.correctControl(rightTorque, leftTorque)

        dX = (self.I + self.A*dt)@np.array([x, dx, theta, dtheta, psi, dpsi]) + self.B*dt@np.array([rightTorque, leftTorque])

        self.X = self.X + dX*dt

    def getPosition(self):
        return self.X[0], self.X[1]
        
        
# test model
robot = Robot()
for i in range(1000):
    robot.updateConfiguration(0.1, 0.1, 0.01)
    print("Time: ",i*0.01," ",robot.getPosition())
    time.sleep(0.05)


import casadi as ca
import numpy as np
import math

def shift(u, x_n):
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return u_end, x_n

class AltitudeMPC:
    def __init__(self, robot, T=0.02, N=30, Q=np.diag([40.0, 1.0]), R=np.diag([1.0])):
        self.robot = robot
        self.T = T  # time step
        self.N = N  # horizon length

        # weight matrix
        self.Q = Q
        self.R = R

        # The history states and controls
        self.next_states = np.zeros((self.N+1, 2))
        self.u0 = np.zeros((self.N, 1))

        self.setupController()
    
    def setupController(self):
        self.opti = ca.Opti()
        # the total thrust
        self.opt_controls = self.opti.variable(self.N, 2)
        thrust = self.opt_controls

        self.opt_states = self.opti.variable(self.N+1, 6)
        x = self.opt_states[:,0]
        dx = self.opt_states[:,1]
        theta = self.opt_states[:,2]
        dtheta = self.opt_states[:,3]
        psi = self.opt_states[:,4]
        dpsi = self.opt_states[:,5]

        # create model
        f = lambda x_, u_: ca.vertcat(*[
            x_[1],
            self.quad.g - u_/self.quad.mq,
        ])

        # parameters, these parameters are the reference trajectories of the pose and inputs
        self.opt_u_ref = self.opti.parameter(self.N, 1)
        self.opt_x_ref = self.opti.parameter(self.N+1, 2)

        # initial condition
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T*self.T
            self.opti.subject_to(self.opt_states[i+1, :] == x_next)
        
        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref[i+1, :]
            control_error_ = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                        + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        # boundary and control conditions
        self.opti.subject_to(self.opti.bounded(-math.inf, z, self.quad.max_z))
        self.opti.subject_to(self.opti.bounded(self.quad.min_dz, dz, self.quad.max_dz))

        self.opti.subject_to(self.opti.bounded(self.quad.min_thrust, thrust, self.quad.max_thrust))

        opts_setting = {'ipopt.max_iter':2000,
                        'ipopt.print_level':0,
                        'print_time':0,
                        'ipopt.acceptable_tol':1e-8,
                        'ipopt.acceptable_obj_change_tol':1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, next_trajectories, next_controls):
        ## set parameter, here only update initial state of x (x0)
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)
        
        ## provide the initial guess of the optimization targets
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0.reshape(self.N, 1))
        
        ## solve the problem
        sol = self.opti.solve()
        
        ## obtain the control input
        u_res = sol.value(self.opt_controls)
        x_m = sol.value(self.opt_states)
        self.u0, self.next_states = shift(u_res, x_m)
        return u_res
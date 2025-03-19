import casadi as ca
import numpy as np
import math

class NMPCController:
    def __init__(self, init_pos, min_vx, max_vx, min_omega, max_omega,
                T=0.02, N=30, Q=np.diag([50.0, 50.0, 10.0]), R=np.diag([1.0, 1.0])):
        self.T = T          # time step
        self.N = N          # horizon length

        self.Q = Q          # Weight matrix for states
        self.R = R          # Weight matrix for controls

        # Constraints
        self.min_vx = min_vx
        self.max_vx = max_vx
        self.min_omega = min_omega
        self.max_omega = max_omega

        self.max_dvx = 0.8
        self.max_domega = math.pi/6
    
        # The history states and controls
        self.next_states = np.ones((self.N+1, 3))*init_pos
        self.u0 = np.zeros((self.N, 2))

        self.setup_controller()
    
    def setup_controller(self):
        self.opti = ca.Opti()

        # state variables: position and orientation
        self.opt_states = self.opti.variable(self.N+1, 3)
        x = self.opt_states[:,0]
        y = self.opt_states[:,1]
        theta = self.opt_states[:,2]

        # control variables
        self.opt_controls = self.opti.variable(self.N, 2)
        vx = self.opt_controls[:,0]    # Fixed indexing
        omega = self.opt_controls[:,1] # Fixed indexing

        # create model
        f = lambda x_, u_: ca.vertcat(*[
            ca.cos(x_[2])*u_[0],  # dx
            ca.sin(x_[2])*u_[0],  # dy
            u_[1],                # dtheta
        ])

        # parameters for reference trajectories
        self.opt_u_ref = self.opti.parameter(self.N, 2)
        self.opt_x_ref = self.opti.parameter(self.N+1, 3)

        # initial condition
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + (self.T * f(self.opt_states[i, :], self.opt_controls[i, :])).T
            self.opti.subject_to(self.opt_states[i+1, :] == x_next)
        
        # cost function
        obj = 0
        for i in range(self.N):
            state_error = self.opt_states[i, :] - self.opt_x_ref[i, :]
            control_error = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([state_error, self.Q, state_error.T]) + \
                  ca.mtimes([control_error, self.R, control_error.T])
        self.opti.minimize(obj)

        # control rate constraints
        for i in range(self.N-1):
            dvel = (self.opt_controls[i+1,:] - self.opt_controls[i,:])/self.T
            self.opti.subject_to(self.opti.bounded(-self.max_dvx, dvel[0], self.max_dvx))
            self.opti.subject_to(self.opti.bounded(-self.max_domega, dvel[1], self.max_domega))

        # control bounds (applied to all timesteps)
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(self.min_vx, self.opt_controls[i,0], self.max_vx))
            self.opti.subject_to(self.opti.bounded(self.min_omega, self.opt_controls[i,1], self.max_omega))

        opts_setting = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }

        self.opti.solver('ipopt', opts_setting)
    
    def solve(self, next_trajectories, next_controls):
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)
        
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)
        
        sol = self.opti.solve()
        
        self.u0 = sol.value(self.opt_controls)
        self.next_states = sol.value(self.opt_states)
        return self.u0[0,:]

class Robot:
    def __init__(self, init_state):
        self.X = np.array(init_state, dtype=float)
    
    def get_state(self):
        return self.X.copy()
    
    def update(self, u, dt):
        dx = np.array([u[0]*np.cos(self.X[2]), 
                      u[0]*np.sin(self.X[2]), 
                      u[1]], dtype=float)
        self.X = self.X + dx * dt
        return self.X

def main():
    # Initialize parameters
    init_pos = np.array([0.0, 0.0, 0.0])  # x, y, theta
    min_vx, max_vx = -1.0, 1.0
    min_omega, max_omega = -math.pi/2, math.pi/2
    T = 0.02
    N = 30

    # Create controller and robot
    controller = NMPCController(init_pos, min_vx, max_vx, min_omega, max_omega, T=T, N=N)
    robot = Robot(init_pos)

    # Create simple circular reference trajectory
    t = np.linspace(0, T*N, N+1)
    radius = 1.0
    ref_states = np.zeros((N+1, 3))
    ref_states[:,0] = 5 # x
    ref_states[:,1] = 5  # y
    ref_states[:,2] = 3.14 # theta
    
    ref_controls = np.zeros((N, 2))
    ref_controls[:,0] = 0.5  # constant linear velocity
    ref_controls[:,1] = 0.1  # constant angular velocity

    n_steps = 10000
    states_history = [robot.get_state()]
    
    for i in range(n_steps):
        ref_states[0, :] = robot.get_state()
        u = controller.solve(ref_states, ref_controls)
        new_state = robot.update(u, T)
        states_history.append(new_state)
        
        print(f"Step {i}: x={new_state[0]:.3f}, y={new_state[1]:.3f}, theta={new_state[2]:.3f}, "
              f"vx={u[0]:.3f}, omega={u[1]:.3f}")

    states_history = np.array(states_history)
    
    # Simple verification
    print("\nSimulation Summary:")
    print(f"Final position: x={states_history[-1,0]:.3f}, y={states_history[-1,1]:.3f}")
    print(f"Distance traveled: {np.sum(np.sqrt(np.sum(np.diff(states_history[:,:2], axis=0)**2, axis=1))):.3f}")

if __name__ == "__main__":
    main()
    



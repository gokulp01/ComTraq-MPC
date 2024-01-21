import numpy as np
from scipy.optimize import minimize
import copy



class ParticleFilter:
    def __init__(self, num_particles, init_state):
        self.num_particles = num_particles
        self.particles = np.empty((num_particles, 3))  # x, y, theta
        self.particles[:, 0] = init_state[0]  # x
        self.particles[:, 1] = init_state[1]  # y
        self.particles[:, 2] = init_state[2]  # theta
        print(self.particles[0])
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, move_x, move_y, rotate_theta, std_pos, std_theta):
        """ Move the particles based on the motion model. """
        self.particles[:, 0] += move_x + np.random.normal(0, std_pos, self.num_particles)
        self.particles[:, 1] += move_y + np.random.normal(0, std_pos, self.num_particles)
        self.particles[:, 2] += rotate_theta + np.random.normal(0, std_theta, self.num_particles)


    def update(self, z, std):
        """ Update particle weights based on measurement. """
        distances = np.linalg.norm(self.particles[:, :3] - z, axis=1)
        self.weights = np.exp(-distances ** 2 / (2 * std ** 2))
        self.weights += 1.e-300  # avoid divide by zero
        self.weights /= sum(self.weights)

    def resample(self):
        """ Resample particles based on weights. """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        """ Estimate the current position and orientation. """
        return np.average(self.particles, weights=self.weights, axis=0)


class Car_Dynamics:
    def __init__(self, x_0, y_0, v_0, psi_0, length, dt, pf):
        self.dt = dt             # sampling time
        self.L = length          # vehicle length
        self.x = pf.estimate()[0]
        self.y = pf.estimate()[1]
        self.v = v_0
        self.psi = pf.estimate()[2]
        self.pf=pf
        self.state = np.array([[self.x, self.y, self.v, self.psi]]).T
        self.state_pf = np.array([[self.x, self.y, self.psi]]).T

    def move(self, accelerate, delta):
        
        x_dot = self.v*np.cos(self.psi)
        # print( self.v)
        y_dot = self.v*np.sin(self.psi)
        v_dot = accelerate
        psi_dot = self.v*np.tan(delta)/self.L
        return np.array([[x_dot, y_dot, v_dot, psi_dot]]).T

    def update_state(self, state_dot, obs_x, obs_y, obs_theta):
        # self.u_k = command
        # self.z_k = state
        # print(state_dot)
        self.pf.predict(state_dot[0,0]*self.dt, state_dot[1,0]*self.dt, state_dot[3,0]*self.dt, 0.01, 0.01)
        self.pf.update(np.array([obs_x, obs_y, obs_theta]), 0.1)
        self.pf.resample()
        # self.pf.estimate()
        self.state=self.state + self.dt*state_dot
        self.state_pf = self.pf.estimate()
        print(self.state, self.state_pf)
        # self.state = self.state + self.dt*state_dot
        self.x = self.state_pf[0]
        self.y = self.state_pf[1]
        self.v = self.state[2,0]
        self.psi = self.state_pf[2]
        # print(self.state)

    
class MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix
        self.num_particles=10000                               


    def mpc_cost(self, u_k, my_car, points):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
    
        desired_state = points.T
        cost = 0.0

        for i in range(self.horiz):
            state_dot = mpc_car.move(u_k[0,i], u_k[1,i])
            mpc_car.update_state(state_dot, my_car.x, my_car.y, my_car.psi)
        
            z_k[:,i] = [mpc_car.x, mpc_car.y] # replace with belief state from particle filter
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]



######################################################################################################################################################################

class Linear_MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix
        self.dt=0.2   
        self.L=4                          

    def make_model(self, v, psi, delta):        
        # matrices
        # 4*4
        A = np.array([[1, 0, self.dt*np.cos(psi)         , -self.dt*v*np.sin(psi)],
                    [0, 1, self.dt*np.sin(psi)         , self.dt*v*np.cos(psi) ],
                    [0, 0, 1                           , 0                     ],
                    [0, 0, self.dt*np.tan(delta)/self.L, 1                     ]])
        # 4*2 
        B = np.array([[0      , 0                                  ],
                    [0      , 0                                  ],
                    [self.dt, 0                                  ],
                    [0      , self.dt*v/(self.L*np.cos(delta)**2)]])

        # 4*1
        C = np.array([[self.dt*v* np.sin(psi)*psi                ],
                    [-self.dt*v*np.cos(psi)*psi                ],
                    [0                                         ],
                    [-self.dt*v*delta/(self.L*np.cos(delta)**2)]])
        
        return A, B, C

    def mpc_cost(self, u_k, my_car, points):
        
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
        desired_state = points.T
        cost = 0.0
        old_state = np.array([my_car.x, my_car.y, my_car.v, my_car.psi]).reshape(4,1)

        for i in range(self.horiz):
            delta = u_k[1,i]
            A,B,C = self.make_model(my_car.v, my_car.psi, delta)
            new_state = A@old_state + B@u_k + C
        
            z_k[:,i] = [new_state[0,0], new_state[1,0]]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
            
            old_state = new_state
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]
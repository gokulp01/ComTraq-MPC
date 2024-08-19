import copy

import numpy as np
from scipy.optimize import minimize


class ParticleFilter:
    def __init__(self, num_particles, init_state):
        self.num_particles = num_particles
        self.particles = np.empty((num_particles, 3))  # x, y, theta
        self.particles[:, 0] = init_state[0]  # x
        self.particles[:, 1] = init_state[1]  # y
        self.particles[:, 2] = init_state[2]  # theta
        # print(self.particles[0])
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, move_x, move_y, rotate_theta, std_pos, std_theta):
        self.particles[:, 0] += move_x + np.random.normal(
            0, std_pos, self.num_particles
        )
        self.particles[:, 1] += move_y + np.random.normal(
            0, std_pos, self.num_particles
        )
        self.particles[:, 2] += rotate_theta + np.random.normal(
            0, std_theta, self.num_particles
        )

    def update(self, z, std):
        distances = np.linalg.norm(self.particles[:, :3] - z, axis=1)
        self.weights = np.exp(-(distances**2) / (2 * std**2))
        self.weights += 1.0e-300  # avoid divide by zero
        self.weights /= sum(self.weights)

    def resample(self):
        indices = np.random.choice(
            self.num_particles, self.num_particles, p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)


class Car_Dynamics:
    def __init__(self, x_0, y_0, v_0, psi_0, length, dt, pf):
        self.dt = dt  # sampling time
        self.L = length  # vehicle length
        self.x = pf.estimate()[0]
        self.y = pf.estimate()[1]
        self.v = v_0
        self.psi = pf.estimate()[2]
        self.pf = pf
        self.x_true = x_0
        self.y_true = y_0
        self.pf_var = np.var(self.pf.particles, axis=0)
        self.del_var = np.zeros(3)
        self.psi_true = psi_0
        self.state = np.array([[self.x_true, self.y_true, self.v, self.psi_true]]).T
        self.state_pf = np.array([[self.x, self.y, self.psi]]).T

    def move(self, accelerate, delta):
        x_dot = self.v * np.cos(self.psi)
        # print( self.v)
        y_dot = self.v * np.sin(self.psi)
        v_dot = accelerate
        psi_dot = self.v * np.tan(delta) / self.L
        return np.array([[x_dot, y_dot, v_dot, psi_dot]]).T

    def update_state(self, state_dot, obs_x, obs_y, obs_theta, best_action):
        slip = np.random.normal(0, 15 * np.pi / 180)
        self.state = self.state + self.dt * state_dot
        if best_action == 1:
            # print("communicate")
            # self.state_pf = self.state.copy()
            self.x = self.state[0, 0]
            self.y = self.state[1, 0]
            self.psi = self.state[3, 0]

            self.pf.particles[:, 0] = self.x
            self.pf.particles[:, 1] = self.y
            self.pf.particles[:, 2] = self.psi
            # print(np.var(self.pf.particles, axis=0))
        else:
            # self.u_k = command
            # self.z_k = state
            # print(state_dot)
            # print("----"*8)
            # print("here")
            # print(f"observed state: {obs_x, obs_y, obs_theta}")

            self.pf.predict(
                state_dot[0, 0] * self.dt,
                state_dot[1, 0] * self.dt,
                state_dot[3, 0] * self.dt,
                0.1,
                0.1,
            )
            self.pf.update(np.array([obs_x, obs_y, obs_theta]), 0.001)
            self.pf.resample()

            # self.pf.estimate()

            self.state_pf = self.pf.estimate()
            # print(f"state_pf: {self.state_pf}")
            # print(f"state: {self.state}")

            self.x = self.state_pf[0]
            self.y = self.state_pf[1]
            self.psi = self.state_pf[2] + slip
        # print("hrere")
        temp = self.pf_var
        self.pf_var = np.var(self.pf.particles, axis=0)
        self.del_var = self.pf_var - temp
        self.v = self.state[2, 0]
        if self.v > 0.2 * 15:
            self.v = 0.2 * 15
        elif self.v < -0.2 * 15:
            self.v = -0.2 * 15

        self.x_true = self.state[0, 0]
        self.y_true = self.state[1, 0]
        self.psi_true = self.state[3, 0] + slip
        # print(f"true xyz state: {self.x_true, self.y_true, self.psi_true}")
        # print("===="*8)
        # print(self.state)

    def update_state_optimize(self, state_dot):
        self.state = self.state + self.dt * state_dot
        self.x = self.state[0, 0]
        self.y = self.state[1, 0]
        self.v = self.state[2, 0]
        self.psi = self.state[3, 0]
        if self.v > 0.2 * 15:
            self.v = 0.2 * 15
        elif self.v < -0.2 * 15:
            self.v = -0.2 * 15


class MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])  # input cost matrix
        self.Rd = np.diag([0.01, 1.0])  # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])  # state cost matrix
        self.Qf = self.Q  # state final matrix
        self.num_particles = 10000

    def mpc_cost(self, u_k, my_car, points):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz + 1))

        desired_state = points.T
        cost = 0.0

        for i in range(self.horiz):
            state_dot = mpc_car.move(u_k[0, i], u_k[1, i])
            mpc_car.update_state_optimize(state_dot)

            z_k[:, i] = [
                mpc_car.x,
                mpc_car.y,
            ]  # replace with belief state from particle filter
            cost += np.sum(self.R @ (u_k[:, i] ** 2))
            cost += np.sum(self.Q @ ((desired_state[:, i] - z_k[:, i]) ** 2))
            if i < (self.horiz - 1):
                cost += np.sum(self.Rd @ ((u_k[:, i + 1] - u_k[:, i]) ** 2))
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-0.2 * 15, 0.2 * 15), (np.deg2rad(-180), np.deg2rad(180))] * self.horiz
        x0 = np.zeros((2 * self.horiz))

        try:
            result = minimize(
                self.mpc_cost,
                args=(my_car, points),
                x0=x0,
                method="SLSQP",
                bounds=bnd,
            )
            return result.x[0], result.x[1], result.fun
        except ValueError as e:
            print(f"An error occurred during optimization: {e}")
            # Handle the error, e.g., by returning a default value or taking corrective action
            return 60, 90, 0

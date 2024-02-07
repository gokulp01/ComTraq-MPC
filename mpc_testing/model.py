import numpy as np
from control import Car_Dynamics, MPC_Controller, ParticleFilter
import gymnasium as gym
from gymnasium import Env, spaces


class USV(Env):
    def __init__(
        self, x, y, psi, v, dt, final_path, path_index, goal, budget, MPC_HORIZON=5
    ):
        self.MPC_HORIZON = MPC_HORIZON
        self.x = x
        self.y = y
        self.path_index = path_index
        self.dt = dt
        self.optimal_path = final_path
        self.goal = goal
        self.controller = MPC_Controller()
        self.car = Car_Dynamics(
            x,
            y,
            v,
            np.deg2rad(psi),
            length=4,
            dt=self.dt,
            pf=ParticleFilter(
                num_particles=1000,
                init_state=np.array([x, y, np.deg2rad(psi)]),
            ),
        )
        self.num_steps = 0
        self.available_budget = budget
        self.initial_budget = budget
        self.num_states = 4
        self.num_actions = 2
        self.action_space = spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),  # Low bounds
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),  # High bounds
            dtype=np.float32,  # It's good practice to define the dtype
        )
        # self.actions = ["communicate", "no_communicate"]
        self.state = [self.car.x, self.car.y, self.car.v, self.car.psi]

    def reset(self, seed=None, options=None):
        self.controller = MPC_Controller()
        self.MPC_HORIZON = 5
        self.num_steps = 0
        self.car = Car_Dynamics(
            60,
            90,
            0,
            np.deg2rad(270),
            length=4,
            dt=self.dt,
            pf=ParticleFilter(
                num_particles=1000,
                init_state=np.array([60, 90, np.deg2rad(270)]),
            ),
        )
        self.path_index = 0
        self.available_budget = self.initial_budget
        info = {"budget": self.available_budget, "path_index": self.path_index}
        return np.array(
            [self.car.x, self.car.y, self.car.psi, self.car.v, self.available_budget],
            dtype=np.float32,
        ), info

    def render(self):
        pass

    def close(self):
        pass

    def waypoint_reached(self):
        if np.linalg.norm(np.array([self.car.x, self.car.y]) - np.array(self.goal)) < 4:
            print("reached waypoint yas")
        return (
            np.linalg.norm(np.array([self.car.x, self.car.y]) - np.array(self.goal)) < 4
        )

    def reward(self):
        reward = 0
        reward -= 1
        # reward -= np.linalg.norm(
        #     np.array([self.car.x, self.car.y]) - np.array(self.goal)
        # ) / np.linalg.norm(np.array([self.x, self.y]) - np.array(self.goal))
        # print(f"reward1: {reward}")
        # reward -= (self.car.del_var[0] + self.car.del_var[1]) * 10
        # print(f"reward2: {np.linalg.norm(self.car.del_var)*10}")
        reward -= np.linalg.norm(
            np.array([self.car.x, self.car.y])
            - np.array([self.car.x_true, self.car.y_true])
        )
        if self.available_budget <= 0:
            reward -= 10
        # if self.waypoint_reached():
        #     reward += 10

        return reward

    def step(self, action):
        # print(f"action: {action}")
        if action == 1:
            self.available_budget -= 1
        acc, delta, cost = self.controller.optimize(
            self.car,
            self.optimal_path[self.path_index : self.path_index + self.MPC_HORIZON],
        )

        self.car.update_state(
            self.car.move(acc, delta), self.car.x, self.car.y, self.car.psi, action
        )
        reward = self.reward()
        done = False
        terminated = False
        truncated = False
        self.path_index += 1
        self.num_steps += 1
        # print(self.num_steps)

        if self.num_steps == len(self.optimal_path):
            truncated = True

        # print(self.available_budget)
        if self.waypoint_reached() or self.available_budget <= 0:
            terminated = True
        done = terminated or truncated
        info = {
            "budget": self.available_budget,
            "path_index": self.path_index,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
        }
        # print(f"info: {info}")
        return (
            np.array(
                [
                    self.car.x,
                    self.car.y,
                    self.car.psi,
                    self.car.v,
                    self.available_budget,
                ],
                dtype=np.float32,
            ),
            reward,
            terminated,
            truncated,
            info,
        )

    def step_test(self, action):
        # print(f"action: {action}")
        if action == 1:
            self.available_budget -= 1
        acc, delta, cost = self.controller.optimize(
            self.car,
            self.optimal_path[self.path_index : self.path_index + self.MPC_HORIZON],
        )

        self.car.update_state(
            self.car.move(acc, delta), self.car.x, self.car.y, self.car.psi, action
        )
        reward = self.reward()
        done = False
        terminated = False
        truncated = False
        self.path_index += 1
        self.num_steps += 1
        # print(self.num_steps)

        if self.num_steps == len(self.optimal_path):
            truncated = True

        if self.waypoint_reached():
            terminated = True
        done = terminated or truncated
        info = {
            "budget": self.available_budget,
            "path_index": self.path_index,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
        }
        # print(f"info: {info}")
        return (
            np.array(
                [
                    self.car.x,
                    self.car.y,
                    self.car.psi,
                    self.car.v,
                    self.available_budget,
                    acc,
                    delta,
                ],
                dtype=np.float32,
            ),
            reward,
            terminated,
            truncated,
            info,
        )


#         self.x, self.y, self.psi, self.v = self.car.update_state(
#             self.car.move(action[0], action[1]),
#             self.optimal_path[0, 0],
#             self.optimal_path[0, 1],
#             angle_of_line(self.optimal_path[0], self.optimal_path[1]),
#             action,
#         )
#         return self.x, self.y, self.psi, self.v

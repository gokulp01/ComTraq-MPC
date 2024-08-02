import numpy as np
from control import Car_Dynamics, MPC_Controller, ParticleFilter
import gymnasium as gym
from gymnasium import Env, spaces
import random


class USV(Env):
    def __init__(
        self, v, dt, path_index, goal, budget, initial_positions, final_paths, MPC_HORIZON=5,
    ):
        self.MPC_HORIZON = MPC_HORIZON
        index = random.choice(range(len(initial_positions)))
        self.x, self.y, self.psi = initial_positions[index]
        self.optimal_path = final_paths[index]
        self.path_index = path_index
        self.dt = dt
        # self.optimal_path = final_path
        self.goal = goal
        self.controller = MPC_Controller()
        self.car = Car_Dynamics(
            self.x,
            self.y,
            v,
            np.deg2rad(self.psi),
            length=0.138*15,
            dt=self.dt,
            pf=ParticleFilter(
                num_particles=1000,
                init_state=np.array([self.x, self.y, np.deg2rad(self.psi)]),
            ),
        )
        self.num_steps = 0
        self.available_budget = budget
        self.initial_budget = budget
        
        self.action_map=np.zeros(10002,dtype=object)
        self.acc_vals=np.linspace(-0.2*15,0.2*15,100)
        self.ste_vals=np.linspace(-np.deg2rad(60),np.deg2rad(60),100)
        for i in range(len(self.acc_vals)):
            for j in range(len(self.ste_vals)):
                self.action_map[i*len(self.ste_vals)+j+2] = (self.acc_vals[i],self.ste_vals[j])
        self.action_space = spaces.Discrete(10002)
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),  # Low bounds
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),  # High bounds
            dtype=np.float32,  # It's good practice to define the dtype
        )
        # self.actions = ["communicate", "no_communicate"]
        self.state = [self.car.x, self.car.y, self.car.v, self.car.psi]
        self.initial_positions = initial_positions
        self.final_paths = final_paths
        self.optimal_path = None
        # self.next_initial_position = None

    # def set_initial_state(self, x, y, psi):
    #     self.next_initial_position = (x, y, psi)

    def reset(self, seed=None, options=None):
        self.controller = MPC_Controller()
        self.MPC_HORIZON = 5
        self.num_steps = 0
        # if self.next_initial_position is not None:
        index = random.choice(range(len(self.initial_positions)))
        self.x, self.y, self.psi = self.initial_positions[index]
        # print(f"initial position: {car_x, car_y, car_psi}")
        self.optimal_path = self.final_paths[index]
        # self.next_initial_position = None
        self.car = Car_Dynamics(
            self.x,
            self.y,
            0,
            np.deg2rad(self.psi),
            length=0.138*15,
            dt=self.dt,
            pf=ParticleFilter(
                num_particles=1000,
                init_state=np.array([self.x, self.y, np.deg2rad(self.psi)]),
            ),
        )
        self.path_index = 0
        self.available_budget = self.initial_budget
        info = {"budget": self.available_budget, "path_index": self.path_index}
        return np.array(
            [self.car.x, self.car.y, self.car.psi, self.car.v, self.available_budget], dtype=np.float32
        ), info

    def render(self):
        pass

    def close(self):
        pass

    def waypoint_reached(self):
        if np.linalg.norm(np.array([self.car.x, self.car.y]) - np.array(self.goal)) < .3*10:
            print("reached waypoint yas")
        return (
            np.linalg.norm(np.array([self.car.x, self.car.y]) - np.array(self.goal)) < .3*10
        )

    def reward(self):
        reward = 0
        # reward -= 1
        reward -= np.linalg.norm(
            np.array([self.car.x, self.car.y]) - self.optimal_path[self.path_index]) 
        # print(f"reward1: {reward}")
        reward -= np.linalg.norm(np.array([self.car.x, self.car.y]) - np.array([self.car.x_true, self.car.y_true]))*10
        # print(f"reward2: {np.linalg.norm(np.array([self.car.x, self.car.y]) - np.array([self.car.x_true, self.car.y_true]))}")
        # reward -= (self.car.del_var[0] + self.car.del_var[1]) * 100
        # print(f"reward2: {np.linalg.norm(self.car.del_var)*10}")
        if self.available_budget <= 0:
            reward -= 100
        # if self.waypoint_reached():
        #     reward += 10

        return reward

    def step(self, action):
        # action = 1
        # print(f"action: {action}")
        # print(f"car: {self.car.x, self.car.y, self.car.psi, self.car.v}")
        # print(f"true: {self.car.x_true, self.car.y_true, self.car.psi_true, self.car.v}")
        # action = 0
        # print(f"error:{np.linalg.norm(np.array([self.car.x, self.car.y]) - np.array([self.car.x_true, self.car.y_true]))}")
        done = False
        terminated = False
        truncated = False
        if action == 1:
            self.available_budget -= 1
            reward = self.reward()
            self.path_index += 1
            self.num_steps += 1
            if self.num_steps == len(self.optimal_path):
                truncated = True

            if self.waypoint_reached():
                terminated = True 
            info = {
                "budget": self.available_budget,
                "path_index": self.path_index,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
            }
            self.car.x=self.car.x_true
            self.car.y=self.car.y_true
            self.car.psi=self.car.psi_true

            return (
            np.array(
                [self.car.x, self.car.y, self.car.psi, self.car.v, self.available_budget],
                dtype=np.float32,
            ),
            reward,
            terminated,
            truncated,
            info,
        )
        elif action == 0:
            reward = self.reward()
            self.path_index += 1
            self.num_steps += 1
            if self.num_steps == len(self.optimal_path):
                truncated = True

            if self.waypoint_reached():
                terminated = True
            info = {
                "budget": self.available_budget,
                "path_index": self.path_index,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
            }
            return (
            np.array(
                [self.car.x, self.car.y, self.car.psi, self.car.v, self.available_budget],
                dtype=np.float32,
            ),
            reward,
            terminated,
            truncated,
            info,
        )
        else:
            acc, delta = self.action_map[action]
            # print(f"acc: {acc}, delta: {delta}")
            self.car.update_state(
                self.car.move(acc, delta), self.car.x, self.car.y, self.car.psi
            )
            reward = self.reward()
            self.path_index += 1
            self.num_steps += 1
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
            return (
                np.array(
                    [self.car.x, self.car.y, self.car.psi, self.car.v, self.available_budget],
                    dtype=np.float32,
                ),
                reward,
                terminated,
                truncated,
                info,
            )


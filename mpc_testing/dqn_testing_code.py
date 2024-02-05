from stable_baselines3 import DQN
from torch import e
from utils import angle_of_line, make_square, DataLogger
import gymnasium
from model import USV
import argparse
import matplotlib.pyplot as plt

# Assuming 'USV' is your custom environment class
# If you're using a Gym environment, replace 'USV' with the appropriate Gym environment ID
import cv2
from model import USV
import numpy as np
import argparse
from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path

from control import Car_Dynamics, MPC_Controller, ParticleFilter

logger = DataLogger()
# Initialize your environment
parser = argparse.ArgumentParser()
parser.add_argument("--x_start", type=int, default=0, help="X of start")
parser.add_argument("--y_start", type=int, default=90, help="Y of start")
parser.add_argument("--psi_start", type=int, default=0, help="psi of start")
parser.add_argument("--x_end", type=int, default=90, help="X of end")
parser.add_argument("--y_end", type=int, default=80, help="Y of end")
parser.add_argument(
    "--parking", type=int, default=1, help="park position in parking1 out of 24"
)

args = parser.parse_args()

start = np.array([args.x_start, args.y_start])
end = np.array([args.x_end, args.y_end])
parking1 = Parking1(args.parking)
end, obs = parking1.generate_obstacles()
env_animate = Environment(obs)

park_path_planner = ParkPathPlanning(obs)
path_planner = PathPlanning(obs)

print("planning park scenario ...")
(
    new_end,
    park_path,
    ensure_path1,
    ensure_path2,
) = park_path_planner.generate_park_scenario(
    int(start[0]), int(start[1]), int(end[0]), int(end[1])
)

print("routing to destination ...")
path = path_planner.plan_path(
    int(start[0]), int(start[1]), int(new_end[0]), int(new_end[1])
)
path = np.vstack([path, ensure_path1])

print("interpolating ...")
interpolated_path = interpolate_path(path, sample_rate=5)
interpolated_park_path = interpolate_path(park_path, sample_rate=2)
interpolated_park_path = np.vstack(
    [ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]]
)


final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

print(len(final_path))
print(new_end)

env = USV(
    x=args.x_start,
    y=args.y_start,
    psi=args.psi_start,
    v=0,
    dt=0.2,
    final_path=final_path,
    path_index=0,
    goal=new_end,
    budget=40,
)
# Load the saved model
model = DQN.load("dqn_communication_optimization")

res = env_animate.render(env.car.x, env.car.y, env.car.psi, 0)
cv2.imshow("environment", res)
key = cv2.waitKey(1)

env_animate.draw_path(interpolated_path)
env_animate.draw_path(interpolated_park_path)

# Run the model
num_episodes = 1  # Set the number of episodes you want to run
# print(env.final_path)
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_rewards = 0
    ep_var = [np.linalg.norm(env.car.pf_var)]
    # print(env.path_index)
    errors = [
        np.linalg.norm(
            np.array([env.car.x, env.car.y]) - final_path[env.path_index][:2]
        )
    ]
    print(errors)
    print("-------")
    communicate_indices = []
    while not done:
        action, _states = model.predict(
            obs[:4], deterministic=True
        )  # Use the model to predict the action
        if action == 1:
            communicate_indices.append(env.path_index)

        print(f"action: {action}")
        obs, rewards, terminated, truncated, info = env.step_test(
            action
        )  # Take the action in the environment
        ep_var.append(np.linalg.norm(env.car.pf_var))
        errors.append(
            np.linalg.norm(
                np.array([env.car.x, env.car.y]) - final_path[env.path_index][:2]
            )
        )
        total_rewards += rewards
        done = truncated or terminated
        print(f"obs: {obs}")
        res = env_animate.render(env.car.x, env.car.y, env.car.psi, obs[-1])

        logger.log(final_path[env.path_index - 1], env.car, obs[-2], obs[-1])

        cv2.imshow("environment", res)
        key = cv2.waitKey(10)
        if key == ord("s"):
            cv2.imwrite("res.png", res * 255)
    print(f"Episode: {episode + 1}, Total Rewards: {total_rewards}")

res = env_animate.render(env.car.x, env.car.y, env.car.psi, 0)
logger.save_data()
cv2.imshow("environment", res)
key = cv2.waitKey()
#############################################################################################
np.save("errors.npy", errors)
np.save("ep_var.npy", ep_var)
np.save("communicate_indices.npy", communicate_indices)
cv2.destroyAllWindows()
env.close()

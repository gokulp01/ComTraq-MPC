import model
import numpy as np
import random
import timeit
import os
import torch


def compress_state(state):
    # Assuming `state` is a 2D array of shape (100000, 5)
    compressed_state = np.mean(state, axis=0)
    return compressed_state


class Simulation_PPO:
    def __init__(self, env, Model, waypoints, opt_epochs, update_policy_timestep, RolloutBuffer, gamma, max_steps, num_states, num_actions):
        self._policy = Model
        self.waypoints=waypoints
        self.env=env
        self._old_policy = Model
        self._old_policy.load_state_dict(self._policy.state_dict())
        self._opt_epochs = opt_epochs
        self._update_policy_timestep = update_policy_timestep
        self._buffer = RolloutBuffer
        self._gamma = gamma
        self._step = 0
        self._max_steps = max_steps
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._return_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._total_steps = []


    def run(self, episode):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        
        print("Simulating...")

        # inits
        s=self.env.reset()
        self.G = 0
        iters = 0
        done=False
        belief=self.env.particles.copy()
        total_reward=0
        steps=0
        
        
        while not done:
            steps += 1

            if steps % self._update_policy_timestep ==0:
                # print(self._buffer.rewards)
                # print(len(self._buffer.rewards))
                print("==Updating actor==")
                self._update()


            # get current state of the intersection
            belief_c=compress_state(belief)

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            
            
            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(belief_c)

            # if the chosen phase is different from the last phase, activate the yellow phase
            next_state, next_belief, reward, done = self.env.step(action, s, self.waypoints)
            self._buffer.rewards.append(reward)

            
            total_reward += reward
            # next_belief_c=compress_state(next_belief)
            # execute the phase selected before
            

            # saving variables for later & accumulate reward
            
            belief = next_belief
            s=next_state

            # saving only the meaningful reward and return to better see if the agent is behaving correctly
            

        # self._save_episode_stats()
        print("Total reward:", total_reward)
        self._reward_store.append(total_reward)
        self._total_steps.append(steps)
        
        simulation_time = round(timeit.default_timer() - start_time, 1)

        # print("Training...")
        # start_time = timeit.default_timer()
        # training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, self._reward_store, self._total_steps
    




    def _choose_action(self, state):
        """
        Choose action according to old policy
        """
        with torch.no_grad():
                state = torch.FloatTensor(state)
                action, action_logprob, state_val = self._old_policy.act(state)
            
        self._buffer.states.append(state)
        self._buffer.actions.append(action)
        self._buffer.logprobs.append(action_logprob)
        self._buffer.state_values.append(state_val)

        return action.item()



    def _update(self):
        """
        Monte Carlo estimate of returns
        """
        rewards = []
        discounted_reward = 0
        for reward in reversed(self._buffer.rewards):
            # if is_terminal:
            #     discounted_reward = 0
            discounted_reward = reward + (self._gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # print(f'Hey2:{rewards}')

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self._buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self._buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self._buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self._buffer.state_values, dim=0)).detach()

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-5)

        for i in range(self._opt_epochs):
            self._policy.train(old_states, old_actions, old_logprobs, rewards, advantages)
        
        self._old_policy.load_state_dict(self._policy.state_dict())

        self._buffer.clear()
  # how much negative reward in this episode
        


    @property
    def reward_store(self):
        return self._reward_store

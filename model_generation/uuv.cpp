#include "uuv.h"
#include "cmath"
#include "config.h"
#include "iostream"
#include "map"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void printVector1(std::vector<double> vec) {
  for (auto i = vec.begin(); i < vec.end(); i++)
    std::cout << *i << " ";
  std::cout << std::endl;
}
// random number generator for int values
int generate_random_int(int min, int max) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> dist(min, max);
  int random_val = dist(rng);
  return random_val;
}

// random number generator for double values
double generate_random_float(double min, double max) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(min, max);
  double random_val = dist(rng);
  return random_val;
}

// for particle filter, we are doing simple random_selection
std::vector<int> weighted_random_selection(int num_particles,
                                           const std::vector<double> &weights) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  std::vector<int> indices;
  for (int i = 0; i < num_particles; ++i) {
    indices.push_back(dist(gen));
  }
  return indices;
}

UUV::UUV() {
  weights.resize(num_particles, 0.0);
  // add all the pvt variables here
}

UUV::~UUV() {
  // destroy all the variables here if needed
}

std::vector<double> UUV::trans_prob_fun(std::vector<double> state, int action,
                                        std::vector<double> waypoints) {
  std::vector<double> next_state;
  std::vector<std::vector<double>> next_state_matrix;
  if (action == 0) {
    next_state_matrix = {
        {state[0] + cos(state[2]), state[1] + sin(state[2]), state[2],
         state[3]},
        {state[0] +
             cos(state[2] + generate_random_float(-M_PI / 12, M_PI / 12)),
         state[1] +
             sin(state[2] + generate_random_float(-M_PI / 12, M_PI / 12)),
         state[2], state[3]},
        {state[0] -
             cos(state[2] - generate_random_float(-M_PI / 12, M_PI / 12)),
         state[1] -
             sin(state[2] - generate_random_float(-M_PI / 12, M_PI / 12)),
         state[2], state[3]}};
  } else if (action == 1) {

    double desired_theta =
        atan2(waypoints[1] - state[1], waypoints[0] - state[0]);
    next_state_matrix = {
        {state[0], state[1], desired_theta, state[3]},
        {state[0], state[1],
         desired_theta + generate_random_float(-M_PI / 12, M_PI / 12),
         state[3]},
        {state[0], state[1],
         desired_theta - generate_random_float(-M_PI / 12, M_PI / 12),
         state[3]}};
  } else {
    next_state_matrix = {{state[0], state[1], state[2], state[3] + comm_cost},
                         {state[0], state[1], state[2], state[3] + comm_cost},
                         {state[0], state[1], state[2], state[3] + comm_cost}};
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(probability_distribution.begin(),
                                 probability_distribution.end());
  int index = d(gen);
  next_state = next_state_matrix[index];
  return next_state;
}

double UUV::reward_fun(std::vector<double> state,
                       std::vector<double> waypoints) {
  double rewards = 0.0;
  rewards =
      sqrt(pow(state[0] - waypoints[0], 2) + pow(state[1] - waypoints[1], 2));
  return rewards;
}

std::vector<double> UUV::obs_fun(std::vector<double> state, int action) {
  std::vector<double> observations;
  if (action == 2) {
    observations = state;
  } else {
    observations = init_state;
  }
  std::cout << "Obs:";
  printVector1(observations);
  return observations;
}

double UUV::observation_prob(std::vector<double> observation,
                             std::vector<double> state, int action) {
  if (observation == state && action == 2) {
    std::cout << "here";
    return 1.0;
  } else if (observation == init_state && action != 2) {
    return 1.0;
  } else
    return 0.0;
}

void UUV::initialize_pf() {
  particles.clear();
  std::vector<double> particle(4);
  for (int i = 0; i < num_particles; i++) {
    particle[0] = generate_random_float(0.0, 6.0);
    particle[1] = generate_random_float(0.0, 6.0);
    particle[2] = generate_random_float(0.0, 6.0);
    particle[3] = generate_random_float(0.0, 60.0);
    particles.push_back(particle);
  }
}

void UUV::predict_pf(int action, std::vector<double> waypoints) {
  for (int i = 0; i < num_particles; i++) {
    std::vector<double> particle = particles[i];
    std::vector<double> next_state =
        trans_prob_fun(particle, action, waypoints);
    particles[i] = next_state;
  }
}

void UUV::update_pf(std::vector<double> observations, int action,
                    std::vector<double> waypoints) {
  predict_pf(action, waypoints);
  for (int i = 0; i < num_particles; i++) {
    weights[i] = observation_prob(observations, particles[i], action);
    // std::cout << weights[i];
  }
  double total_weight = 0.0;
  for (int i = 0; i < num_particles; i++) {
    total_weight += weights[i];
  }
  for (int i = 0; i < num_particles; i++)
    weights[i] = weights[i] / total_weight;
  std::vector<int> indices = weighted_random_selection(num_particles, weights);
  particles = resample_particles(particles, indices);
}

std::vector<std::vector<double>>
UUV::resample_particles(const std::vector<std::vector<double>> &particles,
                        const std::vector<int> &indices) {
  std::vector<std::vector<double>> resampled;
  for (int idx : indices) {
    resampled.push_back(particles[idx]);
  }
  return resampled;
}

std::vector<double> UUV::most_frequent_state() {
  std::map<std::vector<double>, int> state_counts;
  for (const auto &particle : particles) {
    state_counts[particle]++;
  }
  int max_count = -1;
  std::vector<double> max_state;
  for (const auto &[state, count] : state_counts) {
    if (count > max_count) {
      max_count = count;
      max_state = state;
    }
  }
  // printVector(max_state);
  return max_state;
}
std::vector<double> UUV::average_state() {
  std::vector<double> avg_state(4, 0); // Assuming 4 dimensions for the state

  for (const auto &particle : particles) {
    for (int i = 0; i < 4; i++) {
      avg_state[i] += particle[i];
    }
  }

  for (int i = 0; i < 4; i++) {
    avg_state[i] /= num_particles;
  }

  return avg_state;
}
std::tuple<std::vector<double>, std::vector<std::vector<double>>, double, bool>
UUV::step(int action, std::vector<double> s, std::vector<double> waypoints) {
  // cout<<"Input action: "<<action<< endl;
  // printVector(s);
  // printVector(waypoints);
  bool done = false;
  num_steps += 1;
  std::vector<double> belief_state = average_state();
  // cout << "belief state ";
  // printVector(belief_state);
  // int action = distrib(gen);

  // cout << "action " << action << endl;
  std::vector<double> next_state = trans_prob_fun(s, action, waypoints);
  // cout << "next state ";
  // printVector(next_state);
  std::vector<double> observation = obs_fun(next_state, action);
  // cout << "observation ";
  // printVector(observation);
  update_pf(observation, action, waypoints);
  std::vector<std::vector<double>> next_belief = particles;
  std::vector<double> belief_next_state = average_state();
  double reward = reward_fun(next_state, waypoints);
  // cout << "reward: " << reward << endl;
  // cout << "-------------------------------" << endl;
  // cout << "-------------------------------" << endl;

  done = is_terminal_state(next_state, waypoints);

  return {next_state, next_belief, reward, done}; // return as tuple
}
std::vector<double> UUV::reset() {
  std::vector<double> state = init_state;
  num_steps = 0;
  initialize_pf();
  return state;
}

bool UUV::is_terminal_state(std::vector<double> state,
                            std::vector<double> waypoints) {
  // Check if UUV reached the last waypoint
  bool reached_waypoint = sqrt(pow(state[0] - waypoints[0], 2) +
                               pow(state[1] - waypoints[1], 2)) <= 0.5;

  // Check if UUV's communication cost exceeded budget
  bool exceeded_budget = state[3] > budget;

  return reached_waypoint || num_steps >= max_steps;
}

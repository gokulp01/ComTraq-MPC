#include <algorithm>
#include <pybind11/pybind11.h>
#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <ostream>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/stl_bind.h>
using namespace std;

namespace py = pybind11;
// to print vectors
template <typename T> void printVector(const std::vector<T> &vec) {
  for (const auto &item : vec) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

// to generate linspaced_array
vector<double> linspace_test(float start, float end, int step_size) {
  vector<double> linspaced_array((end - start) / step_size + 1);
  for (int i = 0; i <= (end - start) / step_size; i++) {
    linspaced_array[i] = start + step_size * i;
    // cout << linspaced_array[i];
  }
  return linspaced_array;
}

vector<int> linspace_test_int(float start, float end, int step_size) {
  vector<int> linspaced_array((end - start) / step_size + 1);
  for (int i = 0; i <= (end - start) / step_size; i++) {
    linspaced_array[i] = start + step_size * i;
    // cout << linspaced_array[i];
  }
  return linspaced_array;
}

int return_index(vector<vector<int>> vec, vector<int> element) {
  auto it = find(vec.begin(), vec.end(), element);
  if (it != vec.end()) {
    return distance(vec.begin(), it);
  } else {
    return 100000;
  }
}

// UUV class
class UUV {
public:

  int num_particles = 100000; // Number of particles
  vector<vector<double>>
      particles; // The particles representing the belief state
  default_random_engine generator; // Random number generator
  float del_ig;  
  int num_states = 5;  // x,y,z,\psi
  int num_actions = 4; // up, down; theta at which thrust, communication
  int max_steps=100;
  int num_steps=0;
  vector<double> init_state = {0, 0, 0, 0, 0};
  vector<int> action_space = linspace_test_int(0, 3, 1);
  vector<vector<double>> state_history;
  vector<vector<int>> action_history;
  vector<vector<double>> observation_history;
  vector<vector<int>> reward_history;
  // int max_steps = 1000;
  std::vector<double> theta_vals = linspace_test(-M_PI, M_PI, 5);
  std::vector<double> theta_vals_slip = linspace_test(-M_PI/6, M_PI/6, 5);

  float budget = 100.0;
  float init_belief = 1.0;
  double comm_cost = 15.0;
  vector<double> probabilities = {0.8, 0.1, 0.1};

  // Transition probability matrix
  pair<vector<double>, vector<vector<double>>> trans_prob(vector<double> state,
                                                          int action, vector<double> waypoints) {
    vector<vector<double>> next_state;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, M_PI/12);
    int random_index = dist(gen);
    double desired_theta=atan((waypoints[1]-state[1])/(waypoints[0]-state[0]));
    double random_theta = theta_vals_slip[random_index];
    if (action == 0) {
      next_state = {{state[0], state[1], state[2] + 1, state[3], state[4]},
                    {state[0], state[1], state[2] - 1, state[3], state[4]},
                    {state[0] + cos(random_theta), state[1] + sin(random_theta),
                     state[2], state[3] + random_theta, state[4]}};
    } else if (action == 1) {
      next_state = {{state[0], state[1], state[2] - 1, state[3], state[4]},
                    {state[0], state[1], state[2] + 1, state[3], state[4]},
                    {state[0] + cos(random_theta), state[1] + sin(random_theta),
                     state[2], state[3] + random_theta, state[4]}};
    } else if (action == 2) {
      next_state = {{state[0] + cos(desired_theta),
                     state[1] + sin(desired_theta), state[2],
                     desired_theta, state[4]},
                    {state[0] + cos(-random_theta),
                     state[1] + sin(-random_theta), state[2],
                     state[3] - random_theta, state[4]},
                    {state[0] + cos(random_theta), state[1] + sin(random_theta),
                     state[2], state[3] + random_theta, state[4]}};
    } else {
      next_state = {{state[0], state[1], state[2], state[3], state[4]+comm_cost},
                    {state[0], state[1], state[2], state[3], state[4]+comm_cost},
                    {state[0], state[1], state[2], state[3], state[4]+comm_cost}};
      // printVector(next_state[0]);
    }

    std::discrete_distribution<> dist2(probabilities.begin(),
                                       probabilities.end());
    int random_index2 = dist2(gen);
    vector<double> next_state_sel = next_state[random_index2];
    return {next_state_sel, next_state};
  }

  // Observation function
  vector<double> observation_function(vector<double> state, int action) {
    vector<double> observations;
    if (action == 3)
      observations = state;
    else {
      observations = init_state;
    }
    return observations;
  }


float info_gap(vector<vector<double>> particles) {
    std::map<vector<double>, int> state_counts;

    // Count the number of particles in each state
    for (const auto& particle : particles) {
        state_counts[particle]++;
    }

    double entropy = 0.0;

    // Calculate the entropy
    for (const auto& [state, count] : state_counts) {
        double probability = (double)count / num_particles;
        if (probability > 0) {
            entropy -= probability * log(probability);
        }
    }

    return entropy;
}

  // observation probability function

  float observation_prob(vector<double> observation, vector<double> state,
                         int action) {
    if (observation == state && action == 3)
      return 1.0;
    else if (observation == init_state && action != 3)
      return 1.0;
    else
      return 0.0;
  }
  
// waypoint rewards
  float waypoint_reward(vector<double> state, vector<double> waypoints){

    float way_reward = 0.0;
      way_reward = sqrt(pow(state[0] - waypoints[0], 2) +
               pow(state[1] - waypoints[1], 2) +
               pow(state[2] - waypoints[2], 2)); // if the current state is within a sphere of radius 2 --> +3
               // else 0
        
      return way_reward;
  }



  float reward_function(vector<double> state, int action,
                        vector<double> waypoints) {
   float reward=0.0; 
   float w1=-0.7;
   float w2=-0.3;

    if (state[4] > budget) {
      reward -= 100;
    }
    else{
      reward+=w1*waypoint_reward(state, waypoints)+w2*del_ig;
    }
    return reward;
  }

  void initialize_particles() {
    particles.clear();

    for (int i = 0; i < num_particles; i++) {
      vector<double> particle = {0, 0, 0, 0, 0};
      particles.push_back(particle);
    }
  }

  // Inside the UUV class:

  void update_belief(int action, vector<double> observation,vector<double> waypoints) {
    // cout<<"action inside update_bel "<<action<<endl<<"observation inside update_bel ";
    // printVector(observation);
    vector<double> weights(num_particles,
                           1.0); // Initialize weights for each particle to 1.0
    // If the "communicate" action is taken, update the belief based on the
    // observation

    if (action == 3) { // Assuming 74 is your communicate action
      for (int i = 0; i < num_particles; i++) {
        vector<double> particle = particles[i];
        weights[i] = observation_prob(observation, particle, action);
      }

      // Resample particles based on their weights
      discrete_distribution<int> distribution(weights.begin(), weights.end());
      vector<vector<double>> new_particles;

      for (int i = 0; i < num_particles; i++) {
        int index = distribution(generator);
        new_particles.push_back(particles[index]);
      }
      del_ig = abs(info_gap(new_particles)-info_gap(particles));

      particles = new_particles; // Update the particles
    } else {
      // If any action other than "communicate" is taken, propagate particles
      // based on the action
      vector<vector<double>> temp=particles;

      for (int i = 0; i < num_particles; i++) {
        vector<double> particle = particles[i];
        pair<vector<double>, vector<vector<double>>> result =
            trans_prob(particle, action, waypoints);
        particle = result.first;
        particles[i] = particle;

      }
      del_ig=abs(info_gap(particles)-info_gap(temp));
    }
  }

  vector<double> most_frequent_state() {
    map<vector<double>, int> state_counts;
    for (const auto &particle : particles) {
      state_counts[particle]++;
    }

    int max_count = -1;
    vector<double> max_state;
    for (const auto &[state, count] : state_counts) {
      if (count > max_count) {
        max_count = count;
        max_state = state;
      }
    }
    // printVector(max_state);
    return max_state;
  }

std::tuple<vector<double>, vector<vector<double>>, double, bool> step(int action, vector<double> s, vector<double> waypoints){
  // cout<<"Input action: "<<action<< endl;
  // printVector(s);
  // printVector(waypoints);
    bool done=false;
    num_steps+=1;
    vector<double> belief_state = most_frequent_state();
    // cout << "belief state ";
    // printVector(belief_state);
    // int action = distrib(gen);

    // cout << "action " << action << endl;
    pair<vector<double>, vector<vector<double>>> result = trans_prob(s, action, waypoints);
    vector<double> next_state = result.first;
    // cout << "next state ";
    // printVector(next_state);
    vector<double> observation = observation_function(next_state, action);
    // cout << "observation ";
    // printVector(observation);
    update_belief(action, observation, waypoints);
    vector<vector<double>> next_belief=particles;
    vector<double> belief_next_state=most_frequent_state();
    double reward = reward_function(next_state, action, waypoints);  // Assuming reward_function returns double

    // cout << "reward: " << reward << endl; 
    // cout << "-------------------------------" << endl;
    // cout << "-------------------------------" << endl;
      
    done = is_terminal_state(next_state, waypoints);

      //
    return {next_state, next_belief, reward, done};  // return as tuple
}


std::tuple<vector<double>, double, bool> step_rollout(int action, vector<double> s, vector<double> waypoints){
  
    bool done=false;
    pair<vector<double>, vector<vector<double>>> result = trans_prob(s, action, waypoints);
    vector<double> next_state = result.first;
    double reward = reward_function(next_state, action, waypoints);  // Assuming reward_function returns double
      
    done = is_terminal_state(next_state, waypoints);

      //
    return {next_state, reward, done};  // return as tuple
}


vector<double> reset(){
  vector<double> state = init_state;
  num_steps=0;
  initialize_particles();
  return state;
}


bool is_terminal_state(vector<double> state, vector<double> waypoints) {
    // Check if UUV reached the last waypoint
    bool reached_waypoint = sqrt(pow(state[0] - waypoints[0], 2) +
                                 pow(state[1] - waypoints[1], 2) +
                                 pow(state[2] - waypoints[2], 2)) <= 0.5;

    // Check if UUV's communication cost exceeded budget
    bool exceeded_budget = state[4] > budget;

    return reached_waypoint || num_steps >= max_steps;
}

};

// USV class
// class USV {
// public:
//   int num_states = 3; // x,y,\psi
// };


PYBIND11_MODULE(model, m) {
    py::class_<UUV>(m, "UUV")
        .def(py::init<>())
        .def_readwrite("num_states", &UUV::num_states)
        .def_readwrite("num_actions", &UUV::num_actions)
        .def_readwrite("init_state", &UUV::init_state)
        .def_readwrite("action_space", &UUV::action_space)
        // Bind all other variables and methods you want to expose
        .def_readwrite("state_history", &UUV::state_history)
        .def_readwrite("action_history", &UUV::action_history)
        .def_readwrite("observuuvation_history", &UUV::observation_history)
        .def_readwrite("reward_history", &UUV::reward_history)
        .def_readwrite("max_steps", &UUV::max_steps)
        .def_readwrite("theta_vals", &UUV::theta_vals)
        .def_readwrite("theta_vals_slip", &UUV::theta_vals_slip)
        .def_readwrite("budget", &UUV::budget)
        .def_readwrite("init_belief", &UUV::init_belief)
        .def_readwrite("comm_cost", &UUV::comm_cost)
        .def_readwrite("probabilities", &UUV::probabilities)
        .def_readwrite("num_steps", &UUV::num_steps)
        .def_readwrite("particles", &UUV::particles)
        .def("trans_prob", &UUV::trans_prob)
        .def("observation_function", &UUV::observation_function)
        .def("info_gap", &UUV::info_gap)
        .def("observation_prob", &UUV::observation_prob)
        .def("waypoint_reward", &UUV::waypoint_reward)
        .def("reward_function", &UUV::reward_function)
        .def("initialize_particles", &UUV::initialize_particles)
        .def("update_belief", &UUV::update_belief)
        .def("most_frequent_state", &UUV::most_frequent_state)
        .def("is_terminal_state", &UUV::is_terminal_state)
        .def("reset", &UUV::reset)
        .def("step", &UUV::step)
        .def("step_rollout", &UUV::step_rollout);

}

int main() {
  // random_device rd;
  // mt19937 gen(rd());
  // uniform_int_distribution<> distrib(0, 4);
 
  // UUV testing_obj;
  // // printVector(testing_obj.trans_prob({1, 1, 1, 30}, 50));
  // // printVector(testing_obj.trans_prob({1, 1, 1, 30}, 0));
  // // printVector(testing_obj.trans_prob({1, 1, 1, 30}, 1));
  // // printVector(testing_obj.trans_prob({1, 1, 1, 30}, 74));
  // vector<double> s = testing_obj.init_state;
  
  // testing_obj.initialize_particles();
  // vector<double> belief_state=testing_obj.most_frequent_state();
  // vector<double> waypoints={5.0,6.0,7.0};
  // vector<double> next_state;
  // for (int i = 0; i < 100; i++) {
  //   cout<<"s is ";
  //   printVector(s);
  //   cout<<endl;
  //   cout<<"belief is ";
  //   printVector(belief_state);
  //   cout<<endl;
  //   int action=distrib(gen);
  //   cout<<"iteration: "<<i<<endl;
  //   auto [next_state, belief, reward, done] = testing_obj.step(action, s, waypoints);
  //   belief_state=testing_obj.most_frequent_state();
  
  //   s = next_state;
 
  // }
  // printVector(testing_obj.observation_function({1, 1, 1, 30}, 74));
  // printVector(testing_obj.observation_function({1, 1, 1, 30}, 72));
  return 0;
}

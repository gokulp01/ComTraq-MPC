#include <iostream>
#include <random>
#include <vector>
using namespace std;

template <typename T> void printVector(const std::vector<T> &vec) {
  for (const auto &item : vec) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
}

vector<int> linspace_test(float start, float end, int step_size) {
  vector<int> linspaced_array((end - start) / step_size + 1);
  for (int i = 0; i <= (end - start) / step_size; i++) {
    linspaced_array[i] = start + step_size * i;
    // cout << linspaced_array[i];
  }
  return linspaced_array;
}

class UUV {
public:
  int num_states = 4;  // x,y,z,\psi
  int num_actions = 3; // up, down; theta at which thrust, communication
  vector<int> init_state = {0, 0, 0, 0};
  vector<int> action_space = linspace_test(0, 74, 1);
  vector<vector<int>> state_history;
  vector<vector<int>> action_history;
  vector<vector<int>> observation_history;
  vector<vector<int>> reward_history;
  int max_steps = 1000;
  std::vector<int> theta_vals = linspace_test(-180, 180, 5);
  std::vector<int> theta_vals_slip = linspace_test(-30, 30, 5);

  // Transition probability matrix
  vector<int> trans_prob(vector<int> state, int action) {
    vector<vector<int>> next_state;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 15);
    int random_index = dist(gen);
    int random_theta = theta_vals_slip[random_index];
    // cout << random_theta;
    if (action == 0) {
      next_state = {{state[0], state[1], state[2] + 1, state[3]},
                    {state[0], state[1], state[2] - 1, state[3]},
                    {state[0], state[1], state[2], state[3] + random_theta}};
    } else if (action == 1) {
      next_state = {{state[0], state[1], state[2] - 1, state[3]},
                    {state[0], state[1], state[2] + 1, state[3]},
                    {state[0], state[1], state[2], state[3] + random_theta}};
    } else if (action >= 2 && action <= 73) {
      next_state = {{state[0], state[1], state[2], theta_vals[action - 2]},
                    {state[0], state[1], state[2], state[3] - random_theta},
                    {state[0], state[1], state[2], state[3] + random_theta}};
    } else {
      next_state = {{state[0], state[1], state[2], state[3]},
                    {state[0], state[1], state[2], state[3]},
                    {state[0], state[1], state[2], state[3]}};
    }
    vector<double> probabilities = {0.4, 0.3, 0.3};

    std::discrete_distribution<> dist2(probabilities.begin(),
                                       probabilities.end());
    int random_index2 = dist2(gen);
    vector<int> next_state_sel = next_state[random_index2];
    return next_state_sel;
  }
};

// class USV {
// public:
//   int num_states = 3; // x,y,\psi
// };

int main() {

  UUV testing_obj;
  printVector(testing_obj.trans_prob({1, 1, 1, 30}, 50));
  printVector(testing_obj.trans_prob({1, 1, 1, 30}, 0));
  printVector(testing_obj.trans_prob({1, 1, 1, 30}, 1));
  printVector(testing_obj.trans_prob({1, 1, 1, 30}, 74));

  return 0;
}

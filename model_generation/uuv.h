// uuv class
#ifndef UUV_H
#define UUV_H

#include <vector>

void printVector(std::vector<double> vec);
double generate_random_float(double min, double max);
int generate_random_int(int min, int max);
std::vector<int> weighted_random_selection(int num_particles,
                                           const std::vector<double> &weights);
class UUV {

public:
  UUV();
  ~UUV();
  std::vector<double> trans_prob_fun(std::vector<double> state, int action,
                                     std::vector<double> waypoints);

  std::vector<double> obs_fun(std::vector<double> state, int action);

  double reward_fun(std::vector<double> state, std::vector<double> waypoints);

  void initialize_pf();
  void predict_pf(int action, std::vector<double> waypoints);
  void update_pf(std::vector<double> observation, int action,
                 std::vector<double> waypoints);
  double estimate_position_pf();

  double observation_prob(std::vector<double> observation,
                          std::vector<double> state, int action);
  std::vector<std::vector<double>>
  resample_particles(const std::vector<std::vector<double>> &particles,
                     const std::vector<int> &indices);

  std::vector<double> most_frequent_state();
  std::vector<double> average_state();
  std::tuple<std::vector<double>, std::vector<std::vector<double>>, double,
             bool>
  step(int action, std::vector<double> s, std::vector<double> waypoints);
  std::vector<double> reset();
  bool is_terminal_state(std::vector<double> state,
                         std::vector<double> waypoints);

private:
  std::vector<std::vector<double>> particles;
  std::vector<double> weights;
};

#endif // !DEBUG

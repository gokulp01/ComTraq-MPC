#include "config.h"
#include <vector>

int num_states = 4;  // x,y,z,psi
int num_actions = 3; // forward, forward in theta, comm
float budget = 100.0;
double comm_cost = 15.0;
std::vector<double> init_state = {0, 0, 0, 0};
std::vector<double> probability_distribution = {0.8, 0.1, 0.1};
int num_particles = 10000;
int num_steps = 0;
int max_steps = 1000;

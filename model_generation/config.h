// config.h
#ifndef CONFIG_H
#define CONFIG_H
#include <vector>

extern int num_states;
extern int num_actions;
extern std::vector<double> init_state;
extern float budget;
extern double comm_cost;
extern std::vector<double> probability_distribution;
extern int num_particles;
extern int num_steps;
extern int max_steps;
#endif // CONFIG_H

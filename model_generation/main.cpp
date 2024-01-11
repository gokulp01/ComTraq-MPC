#include "config.h"
#include "iostream"
#include "uuv.h"
#include <random>
#include <vector>

using namespace std;

void printVector(vector<double> vec) {
  for (auto i = vec.begin(); i < vec.end(); i++)
    cout << *i << " ";
  cout << endl;
}

int main() {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> distrib(0, 1);

  UUV testing_obj;
  vector<double> s = init_state;
  //
  printVector(s);
  testing_obj.initialize_pf();
  vector<double> belief_state = testing_obj.average_state();
  printVector(belief_state);
  vector<double> waypoints = {5.0, 6.0};
  vector<double> next_state;
  for (int i = 0; i < 50; i++) {
    belief_state[3] = s[3];
    cout << "s is ";
    printVector(s);
    cout << endl;
    cout << "belief is ";
    printVector(belief_state);
    cout << endl;
    int action;
    if (i <= 18)
      action = distrib(gen);
    else
      action = 2;
    cout << "action is " << action << endl;
    cout << "iteration: " << i << endl;
    auto [next_state, belief, reward, done] =
        testing_obj.step(action, s, waypoints);
    belief_state = testing_obj.average_state();
    cout << "Next state: ";
    printVector(next_state);
    cout << "next_belief: ";
    printVector(belief_state);
    s = next_state;
    cout << "-------------------------------------------------" << endl;
  }
  return 0;
}

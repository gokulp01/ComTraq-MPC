#include <iostream>
#include <vector>

using namespace std;

class UUV{
    public:
        int num_states=4;//x,y,z,\psi
        int num_actions=3;//up, down; theta at which thrust, communication
        vector<int> init_state = {0,0,0,0};
        vector<int> action_space = {0,1,2};//0->up/down, 1->theta action, 2->inspect
        vector<vector<int>> state_history;
        vector<vector<int>> action_history;
        vector<vector<int>> observation_history;
        vector<vector<int>> reward_history;
        int max_steps=1000;

        vector<float> trans_prob(state, action){

        }

};


class USV{
    public:
        int num_states=3;//x,y,\psi
};



int main(){
    

    return 0;
}

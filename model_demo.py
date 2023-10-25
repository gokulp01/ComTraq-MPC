# %%
import model

# %%
test=model.UUV()
print("here")
# %%
test.initialize_particles()
print("here2")
# %%
import random
s=test.reset()
b = test.most_frequent_state()
done=False
waypoints = [10,10,10]
while not done:
    print(f'state is {s}, belief is {b}')
    a=random.randint(0,test.num_actions)
    print(f"a is {a}")
    next_state, next_belief, reward, done = test.step(a,s,waypoints)
    next_belief_state=test.most_frequent_state()
    print(f'next state is {next_state}, next_belief is {next_belief_state}, reward is {reward}')
    s = next_state
    b = next_belief_state


      

# %%




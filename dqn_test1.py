import model
import random

def main():
    print("new")
    try:
        
        # Create an instance of your UUV class
        uuv = model.UUV()
        uuv.initialize_particles()
        state = [0.0, 0.0, 0.0, 0.0, 0.0]  # A list of floats
        waypoints = [10.0, 20.0, 30.0]  # A list of floats
        done=False
        while(not done):
        # Dummy data for the example
            action = random.randint(0, 74)  # Assuming the action is an integer between 0 and 74

            # Call the step function and handle the returned results
            next_state, belief_next_state, reward, done = uuv.step(action, state, waypoints)

            # Display the results
            print("Next State:", next_state)
            print("Belief Next State:", belief_next_state)
            print("Reward:", reward)
            print("Done:", done)
            state=next_state
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

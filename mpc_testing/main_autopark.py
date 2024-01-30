import cv2
import numpy as np
from time import sleep
import argparse

from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path
from control import Car_Dynamics, MPC_Controller, ParticleFilter
from utils import angle_of_line, make_square, DataLogger
from ilp import *



def waypoint_reached(car, waypoint):
    if np.linalg.norm(np.array([car.x, car.y]) - waypoint) < 1:
        return True
    return False

budget = 10
comm_counter=0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=90, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')
    parser.add_argument('--parking', type=int, default=1, help='park position in parking1 out of 24')

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables ################################################
    start = np.array([args.x_start, args.y_start])
    end   = np.array([args.x_end, args.y_end])
    #############################################################################################

    # environment margin  : 5
    # pathplanning margin : 5

    ########################## defining obstacles ###############################################
    parking1 = Parking1(args.parking)
    end, obs = parking1.generate_obstacles()

    # add squares
    # square1 = make_square(10,65,20)
    # square2 = make_square(15,30,20)
    # square3 = make_square(50,50,10)
    # obs = np.vstack([obs,square1,square2,square3])

    # Rahneshan logo
    # start = np.array([50,5])
    # end = np.array([35,67])
    # rah = np.flip(cv2.imread('READ_ME/rahneshan_obstacle.png',0), axis=0)
    # obs = np.vstack([np.where(rah<100)[1],np.where(rah<100)[0]]).T

    # new_obs = np.array([[78,78],[79,79],[78,79]])
    # obs = np.vstack([obs,new_obs])
    #############################################################################################

    ########################### initialization ##################################################
    env = Environment(obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2, pf=ParticleFilter(num_particles=1000, init_state=np.array([start[0], start[1], np.deg2rad(args.psi_start)])))    
    MPC_HORIZON = 5
    controller = MPC_Controller()
    # controller = Linear_MPC_Controller()

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ############################# path planning #################################################
    park_path_planner = ParkPathPlanning(obs)
    path_planner = PathPlanning(obs)

    print('planning park scenario ...')
    new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    
    print('routing to destination ...')
    path = path_planner.plan_path(int(start[0]),int(start[1]),int(new_end[0]),int(new_end[1]))
    path = np.vstack([path, ensure_path1])

    print('interpolating ...')
    interpolated_path = interpolate_path(path, sample_rate=5)
    interpolated_park_path = interpolate_path(park_path, sample_rate=2)
    interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])

    env.draw_path(interpolated_path)
    env.draw_path(interpolated_park_path)

    final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

    #############################################################################################
    print(len(final_path))
    print(new_end)
    ################################## control ##################################################
    print('driving to destination ...')
    # state = State(budget, belief=[my_car.x, my_car.y], psi=my_car.psi, velocity=my_car.v, final_path=final_path, path_index=0, waypoint=new_end, del_var_particles=my_car.del_var)
    # best_action = "not_communicate"
    for i,point in enumerate(final_path):
        # state.path_index=i
        # print(f"del variance {state.del_var_particles}")
        if i ==0:
            best_action = "not_communicate"
        else:

            # current_step, total_steps, current_budget, cost_per_communication, current_window_cost
            best_action = make_decision(i, budget, 1, cost, my_car.pf_var[0], my_car.pf_var[1], 1)
        
        acc, delta, cost = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
        my_car.update_state(my_car.move(acc,  delta), my_car.x, my_car.y, my_car.psi, best_action)
        # print(f"here: {my_car.x_true, my_car.y_true, my_car.psi_true}")
        # print(acc)
        res = env.render(my_car.x, my_car.y, my_car.psi, delta)

        logger.log(point, my_car, acc, delta)
        # print(my_car.x, my_car.y)
        print("Best action up:", best_action)
        # print(f"reward: {state.calculate_reward(best_action)}")
        print("---------------")
        if best_action=="communicate":
            comm_counter+=1
            my_car.x=my_car.x_true
            my_car.y=my_car.y_true
            my_car.psi=my_car.psi_true
            budget-=1 
        print(f"communication:{comm_counter}")
        # state.belief = [my_car.x, my_car.y]
        # state.velocity = my_car.v  
        # state.psi = my_car.psi 
        # # print(f"belief main {state.belief}")
        # state.del_var_particles = my_car.del_var
        
        # print(f"variance2 {state.var_particles}")    
        cv2.imshow('environment', res)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('res.png', res*255)

        
        if comm_counter == budget:
            break

    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    #############################################################################################

    cv2.destroyAllWindows()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Details:  This script is used to demonstrate the usage of surrogate models
#           using the Furuta pendulum.
#           The Furuta pendulum is originally introduced in 
#           https://ieeexplore.ieee.org/document/239008/ by Furuta et al..
# Author:   Michał Ślusarczyk based on code by Sebastian Weyrer and Jakob Pflugbeil
# Date:     2026-02-23
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++y++++++++++++++++++++++++++
#pip instal torch
import sys
import time
import os
from models import *
import numpy as np
import matplotlib.pyplot as plt
# add parent directory to find special class
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from furutaPendulum import FurutaPendulum

# define some geometries of the pendulum you want
massArm = 1                 # [kg] mass of arm
massLink = 1                # [kg] mass of link
lengthArm = 1               # [m] length of arm
lengthLink = 1              # [m] length of link
simulationTime = 2  # should be set between 1 and 2
stepSize = 1e-2
np.random.seed(6)
def save_as_csv_file(numpy_array,path_local=r"trajectory.csv"):
    np.savetxt(path_local, numpy_array, delimiter=",")

# load the furuta pendulum class with automatic EOMs import
pendulum = FurutaPendulum(massArm=massArm, massLink=massLink, lengthArm=lengthArm, lengthLink=lengthLink)
def generate_next_state(x0,dt=stepSize):
    t_span=np.array([0,dt])
    data=pendulum.GenerateTrajectory(x0=x0, simulationTime=simulationTime, stepSize=stepSize)
    data_t = data.T
    print(f"traj{data_t[-1]}")
def generate_trajectory_furuta_data(n_trajectories = 100):
    # now you can loop over different initial conditions to generate data
    X_data = []  # Initial states
    Y_data = []  # States after dt
    for i in range(n_trajectories):
        # x = [theta, theta_t, phi, phi_t]
        x0 = np.array([
            wrap(np.random.uniform(-1.0, 1.0)),  # index 0, wrapping due angle equality
            np.random.uniform(-2.0, 2.0),  # index 1
            wrap(np.random.uniform(-1.0, 1.0)),  # index 2
            np.random.uniform(-2.0, 2.0)  # index 3
        ])
        # data is 4 x numberOfTimeSteps (= simulationTime/stepSize)
        #print(x0)
        data = pendulum.GenerateTrajectory(x0=x0, simulationTime=simulationTime, stepSize=stepSize)
        data_t=data.T
        for i in range(len(data_t)-1):
            X_data.append(data_t[i])
            Y_data.append(data_t[i+1])
    return np.array(X_data), np.array(Y_data)

def dynamics(t, x):
    dx = np.zeros(4)
    dx[0] = x[1]
    dx[1] = pendulum.theta_tt(x[0], x[1], x[2], x[3],0,0)
    dx[2] = x[3]
    dx[3] = pendulum.phi_tt(x[0], x[1], x[2], x[3],0,0)
    return dx
def generate_random_data_points_dynamic(n_samples=10000):
    data_train=[]
    X_data = []  # Initial states
    Y_data = []  # dstate
    for i in range(n_samples):
        vector = np.array([
            wrap(np.random.uniform(-1.0, 1.0)),
            np.random.uniform(-2.0, 2.0),
            wrap(np.random.uniform(-1.0, 1.0)),
            np.random.uniform(-2.0, 2.0)
        ])
        sampled_vector = dynamics(0, vector)
        X_data.append(vector)
        Y_data.append(sampled_vector)

    return np.array(X_data), np.array(Y_data)
def plotting(losses,t_span,t_test,data_t,traj_surrogate):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Training loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # Angle comparison
    axes[1, 0].plot(t_test, data_t[:, 0], 'b-', label='RADAU (ground truth)', linewidth=2)
    axes[1, 0].plot(t_test, traj_surrogate[:, 0], 'r--', label='Surrogate', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angle θ (rad)')
    axes[1, 0].set_title('Angular Position')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Angle comparison
    axes[1, 1].plot(t_test, data_t[:, 2], 'b-', label='RADAU (ground truth)', linewidth=2)
    axes[1, 1].plot(t_test, traj_surrogate[:, 2], 'r--', label='Surrogate', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angle θ (rad)')
    axes[1, 1].set_title('Angular Position')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Angular velocity comparison
    axes[2, 0].plot(t_test, data_t[:, 1], 'b-', label='RADAU', linewidth=2)
    axes[2, 0].plot(t_test, traj_surrogate[:, 1], 'r--', label='Surrogate', linewidth=2)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Angular velocity ω (rad/s)')
    axes[2, 0].set_title('Angular Velocity')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Angular velocity comparison
    axes[2, 1].plot(t_test, data_t[:, 3], 'b-', label='RADAU', linewidth=2)
    axes[2, 1].plot(t_test, traj_surrogate[:, 3], 'r--', label='Surrogate', linewidth=2)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Angular velocity ω (rad/s)')
    axes[2, 1].set_title('Angular Velocity')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Phase portrait
    axes[0, 1].plot(data_t[:, 0], data_t[:, 1], 'b-', label='RADAU', linewidth=2)
    axes[0, 1].plot(traj_surrogate[:, 0], traj_surrogate[:, 1], 'r--', label='Surrogate', linewidth=2)
    axes[0, 1].set_xlabel('Angle θ (rad)')
    axes[0, 1].set_ylabel('Angular velocity ω (rad/s)')
    axes[0, 1].set_title('Phase Portrait')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def run_training_next_states(n_trajectories = 1000,epochs=1000):
    x_data, y_data = generate_trajectory_furuta_data(n_trajectories=n_trajectories)
    # x_data,y_data=generate_random_data_points_dynamic(n_samples=1000000)
    print(x_data)
    print(y_data)
    print("in")
    # if you have a new model it should be defined here so it can run
    model_func = lambda: PendulumSurrogate(4,4)
    #model_func = lambda: DeepFeedForwardNet(4,4)
    #model_func = lambda: PendulumSurrogate(4, 4, hidden_size=128)
    model, losses = train_surrogate_simple_val(x_data, y_data,
                                               model_=model_func, epochs=epochs,
                                               lr=0.0005)  # model_=lambda: CNNAutoencoder(n_in=4, n_out=4, latent_dim=8)


    print("\nTesting on new trajectory...")
    # first start_point
    y0 = [0, 0.0, np.pi+0.2, 0]#don't change
    data = pendulum.GenerateTrajectory(x0=y0, simulationTime=simulationTime, stepSize=stepSize,
                                       plot=False)

    t_span = (0, simulationTime)
    t_test = np.linspace(*t_span, int(simulationTime / stepSize))
    traj_surrogate = solve_pendulum_surrogate(model, y0, [0, simulationTime], t_test)
    save_as_csv_file(traj_surrogate, "traj_surrogate.csv")
    print(data.T)
    data_t = data.T
    plotting(losses,t_span,t_test,data_t,traj_surrogate)


    # Calculate error
    error = np.abs(data_t - traj_surrogate)
    print(f"\nMean absolute error - Angle: {np.mean(error[:, 0]):.6f} rad")
    print(f"Mean absolute error - Velocity: {np.mean(error[:, 1]):.6f} rad/s")
    print(f"Max absolute error - Angle: {np.max(error[:, 0]):.6f} rad")
    print(f"Max absolute error - Velocity: {np.max(error[:, 1]):.6f} rad/s")
    plt.show()
    # first start_point
    y0 = [0, 0.1, np.pi-0.4, 0]#don't change
    data = pendulum.GenerateTrajectory(x0=y0, simulationTime=simulationTime, stepSize=stepSize,
                                       plot=False)

    t_span = (0, simulationTime)
    t_test = np.linspace(*t_span, int(simulationTime / stepSize))
    traj_surrogate = solve_pendulum_surrogate(model, y0, [0, simulationTime], t_test)
    save_as_csv_file(traj_surrogate, "traj_surrogate_2.csv")
    print(data.T)
    data_t = data.T
    plotting(losses, t_span, t_test, data_t, traj_surrogate)

    should_the_model_be_saved(model)

def run_dynamics_surrogate(n_points=10000,epochs=1000):
    x_data, y_data = generate_random_data_points_dynamic(n_samples=n_points)
    # if you have a new model it should be defined here so it can run
    model_func = lambda: PendulumSurrogate(4,4)
    model, losses = train_surrogate_simple_val(x_data, y_data,
                                               model_=model_func, epochs=epochs,
                                               lr=0.0005)  # model_=lambda: CNNAutoencoder(n_in=4, n_out=4, latent_dim=8)
    print("\nTesting on new trajectory...")
    # First example
    y0 = [0, 0.0, np.pi+0.2, 0]#don't change
    data = pendulum.GenerateTrajectory(x0=y0, simulationTime=simulationTime, stepSize=stepSize,
                                       plot=True)

    t_span = (0, simulationTime)
    t_test = np.linspace(*t_span, int(simulationTime / stepSize))
    traj_surrogate = solve_pendulum_surrogate(model, y0, [0, simulationTime], t_test)
    save_as_csv_file(traj_surrogate, "traj_surrogate.csv")
    print(data.T)
    data_t = data.T

    plotting(losses, t_span, t_test, data_t, traj_surrogate)

    # Calculate error
    error = np.abs(data_t - traj_surrogate)
    print(f"\nMean absolute error - Angle: {np.mean(error[:, 0]):.6f} rad")
    print(f"Mean absolute error - Velocity: {np.mean(error[:, 1]):.6f} rad/s")
    print(f"Max absolute error - Angle: {np.max(error[:, 0]):.6f} rad")
    print(f"Max absolute error - Velocity: {np.max(error[:, 1]):.6f} rad/s")

    # Second example
    y0 = y0 = [0, 0.1, np.pi-0.4, 0] #don't change
    data = pendulum.GenerateTrajectory(x0=y0, simulationTime=simulationTime, stepSize=stepSize,
                                       plot=True)
    t_span = (0, simulationTime)
    t_test = np.linspace(*t_span, int(simulationTime / stepSize))
    traj_surrogate = solve_pendulum_surrogate(model, y0, [0, simulationTime], t_test)
    save_as_csv_file(traj_surrogate,"traj_surrogate_2.csv")
    print(data.T)
    data_t = data.T
    plotting(losses, t_span, t_test, data_t, traj_surrogate)
    should_the_model_be_saved(model)
    return

def should_the_model_be_saved(model):
    save_path = "model.pth"
    answer = input("\nDo you want to save the model? (y/n): ").strip().lower()

    if answer == "y":
        torch.save(model.state_dict(), save_path)
        print("Model saved to 'model.pth'")
    else:
        print("Model was not saved.")

def main():
    run_next_state = True
    if run_next_state:
        run_training_next_states(1000)
    else:
        run_dynamics_surrogate(1000000)
if __name__ == '__main__':
   main()


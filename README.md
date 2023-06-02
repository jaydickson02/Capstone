# AI Methods for DSS

This is a repository containing the work for a Capstone project undertaken at RMIT by Jay Dickson and Jeremy Bloor.

- Deep Q-Learning
- Particle Swarm Optimisation
- PPO
- Genetic Algorithm NN

## Deep Q-Learning

The Repository is currently setup with a DQN Algorithm and a simple environment.

## Files

    DQN.py - This holds the definition for the Deep Q Network Learning Model and Algorithm.

    Environment.py - Defines the environment and manages stepping through states, calculating rewards and terminating a given sequence.

    Planet.py - Holds the definition for a central gravitational body. Simplistic in nature currently.

    Satellite.py - Holds the definition for a orbital satellite. Contains methods for a number of orbital mathmatics. 

    Main.py - Instantiates the Environment, Planet, Satellite and DQN_Agent. Manages the looping and passes the hyperparameters. Designed to be used from the commandline.

    utils.py - Some helper utilitys mostly for vector math.

    run.sh - Helper function to auto repeat training. Helps avoid crashes. Not needed.

## Dependancies

- Git (I also recommend getting GitHub Desktop for simplicity)
- Needs Python Version 3
- Pip Version 3

To install Dependancies. Navigate to the root of the folder on the commandline.

    pip3 install tensorflow
    pip3 install numpy
    pip3 install pygame
    pip3 install gym
    pip3 install tqdm

## Running the code

Note: This only works on Windows machines or Macs with Intel Chips due to the CPU instructuion set used by Tensorflow.

Training will run the algorithm and save the model as it goes to the best_model.h5 file. It will also track the number of episodes it has completed.

Testing the model will take the best agent and render a simulation of it attempting the problem. This will run indefinitly repeating if the model fails or the runtime reached.

### To Train

    python3 main.py --train --batch_size=10 --episodes=200 --runtime=1500

### To Test

    python3 main.py --test --runtime=2000


## The Current Environment

Currently the model is simulating a single satellite orbiting a single planet. 

- Gravity is modeled as a vector maths implimentation of Newtonian gravity.
- It is a highly simplified model. Some things not accounted for are:
    - Air resistance
    - Pertubations
    - Sightlines
    - Relativity
    - Communication Delay
    - Error around position information

- The model is attempting to create a circular orbit and starts of in a state with a random position and velocity. Fuel is also modelled and drops by a set amount everytime the action taken is to accelerate.

- The model is rewarded and punished as depicted below.
    - +10 for maintaining a rate of change for the altitude.
    - -1 for large altitude rate of changes (> 10).
    - -100 for collision with the planet or walls of the simulation, this will also cause the given sim to terminate.
    - -0.1 for any fuel use.
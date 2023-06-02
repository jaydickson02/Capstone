import time
import os

import contextlib
import io


def speedTestDQN(iterations):

    # Create DQN
    from DQN import DQN
    from Environment import Environment
    from Planet import Planet
    from Satellite import Satellite
    from collections import deque

    # Create objects
    planet = Planet([500, 500], 20, 1, [0, 0, 0])
    satellite = Satellite([500, 500], [0, 0], 3, [0, 0, 0])

    # Create environment
    with contextlib.redirect_stdout(io.StringIO()):
        env = Environment(planet, satellite, 100, 1000, False)
    state = env.reset()

    # Memory
    memory = deque(maxlen=2000)

    dqn = DQN(env, 0.01, 0.95, 1, 0.9995, 0.1, memory)

    # Test the replay method
    start = time.time()
    for i in range(iterations):
        dqn.replay(10)
    end = time.time()

    # Clear the screen
    os.system("clear")

    print("Testing DQN methods..." + "\n")

    print("DQN replay method: " + str(end - start))

    # Test the act method
    start = time.time()
    for i in range(iterations):
        dqn.act(state)
    end = time.time()

    print("DQN act method: " + str(end - start))

    # Test the remember method
    start = time.time()
    for i in range(iterations):
        dqn.remember(state, 1, 1, state, False)
    end = time.time()

    print("DQN remember method: " + str(end - start) + "\n")


def SpeedTestSatellite(iterations):

    # Test Satellite Methods
    from Satellite import Satellite
    from Planet import Planet

    # Create an instance of the satellite and planet
    sat = Satellite([1, 1], [0, 0], 10)
    planet = Planet([0, 0], 100, 1, [0, 0, 0])

    # Test the run method
    start = time.time()
    for i in range(iterations):
        sat.run(planet, 1, 0.01)
    end = time.time()
    print("Satellite run method: " + str(end - start))

    # Test the gravity acceleration method
    start = time.time()
    for i in range(iterations):
        sat.gravityAcceleration(planet, 1)
    end = time.time()
    print("Satellite gravityAcceleration method: " + str(end - start))

    # Test the apply acceleration method
    start = time.time()
    for i in range(iterations):
        sat.applyAcceleration(planet, [0, 0], 0.01, "prograde")
    end = time.time()
    print("Satellite applyAcceleration method: " + str(end - start))

    # Test the propogate method
    start = time.time()
    for i in range(iterations):
        sat.propogate([0, 0])
    end = time.time()
    print("Satellite propogate method: " + str(end - start) + "\n")


def speedTestEnvironment(iterations):

    from Environment import Environment
    from Planet import Planet
    from Satellite import Satellite

    # Create objects
    planet = Planet([500, 500], 20, 1, [0, 0, 0])
    satellite = Satellite([500, 500], [0, 0], 3, [0, 0, 0])

    # Create environment
    env = Environment(planet, satellite, 100, 1000, False)

    # Test the reset method
    start = time.time()
    for i in range(iterations):
        env.reset()
    end = time.time()

    print("Environment reset method: " + str(end - start))

    # Test the next method
    start = time.time()
    for i in range(iterations):
        env.next(1)
    end = time.time()

    print("Environment next method: " + str(end - start))

    # Test the reward method
    start = time.time()
    for i in range(iterations):
        env.reward()
    end = time.time()

    print("Environment reward method: " + str(end - start) + "\n")


# Test all

# Clear the screen
os.system("clear")

print("Testing DQN methods..." + "\n")
speedTestDQN(100000)

print("Testing Satellite methods..." + "\n")
SpeedTestSatellite(100000)

print("Testing Environment methods..." + "\n")
speedTestEnvironment(100000)

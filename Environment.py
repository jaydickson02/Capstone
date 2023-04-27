import utils
import pygame
import numpy as np
from gym import spaces


class Environment:

    # Constructor
    def __init__(self, Planet, Satellite, G, runtime, renderEnv=False):
        self.Planet = Planet
        self.Satellite = Satellite
        self.G = G
        self.renderEnv = renderEnv
        self.runtime = runtime

        # Space Variables
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
                                            high=np.array(
                                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)

        # Tracking variables
        self.previousAltitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)
        self.previousFuel = self.Satellite.fuel
        self.PreviousAltitudeDifference = 1

        # Step Tracking
        self.step = 0

        # Render
        if renderEnv:
            pygame.init()
            self.canvas = pygame.display.set_mode((1000, 1000))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Env Simulation")

        # Altitude Difference List
        self.altitudeDifferenceList = []

    def next(self, action):

        actionValue = [0.001, 0.01, 0, -0.01, -0.001][action]

        self.Satellite.run(self.Planet, self.G, actionValue)

        velocity = self.Satellite.velocity
        position = self.Satellite.position
        altitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)
        fuel = self.Satellite.fuel

        state = [velocity[0], velocity[1],
                 position[0], position[1], altitude, fuel]

        reward, done = self.reward()

        self.step += 1
        # done = False
        if (self.renderEnv):
            self.render(done)

        return (state, reward, done)

    def reward(self):

        reward = 0
        done = False

        altitudeDifference = utils.magnitude(
            (self.Satellite.position - self.Planet.position)) - self.previousAltitude

        self.altitudeDifferenceList.append(altitudeDifference)

        if (altitudeDifference < self.PreviousAltitudeDifference * 0.8):
            reward += 10

        # Check the fuel value compared to the previous fuel value
        if (self.Satellite.fuel < self.previousFuel):
            reward += -0.1

        # Calculate collisions
        if (utils.magnitude(self.Satellite.position - self.Planet.position) < self.Planet.size):
            reward += -100
            done = True

        # Calculate if the satellite has escaped
        if (utils.magnitude(self.Satellite.position - self.Planet.position) > 500):
            reward += -100
            done = True

        # Check if the altitude difference is dramatically increasing
        if (altitudeDifference > 10):
            reward += -1

        if (self.step >= self.runtime):
            done = True

        # Update the previous values
        self.previousAltitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)
        self.previousFuel = self.Satellite.fuel
        self.PreviousAltitudeDifference = altitudeDifference

        # Return the reward and done values
        return (reward, done)

    # Render

    def render(self, done):
        if (self.render and not done):
            self.canvas.fill((255, 255, 255))
            self.Planet.render(self.canvas, 1)
            self.Satellite.render(self.canvas, 1)

            # Update the display
            pygame.display.update()

    def reset(self):
        # Get a random position and velocity for the satellite within bounds
        velocityBounds = [-1, 1]

        velocity = [np.random.uniform(velocityBounds[0], velocityBounds[1]), np.random.uniform(
            velocityBounds[0], velocityBounds[1])]

        # Position is a random vector from the center of the planet
        position = utils.normalise(
            np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]))

        # Scalled randomly to be bigger than the planet but less than the screen size
        position = position * (self.Planet.size + (np.random.uniform(10, 800)))

        # Add the positions to make the planet the origin
        position = position + self.Planet.position

        self.Satellite.position = np.array(position)
        self.Satellite.velocity = np.array(velocity)
        self.Satellite.fuel = 100

        self.step = 0

        # Calculate the initial altitude
        altitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)

        return [velocity[0], velocity[1], position[0], position[1], altitude, 100]

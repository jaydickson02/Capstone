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
        self.altitudeTarget = 0
        self.rewardValue = 0

        # Space Variables
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0]),
                                            high=np.array(
                                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                                            dtype=np.float32)

        # Tracking variables
        self.previousFuel = self.Satellite.fuel

        # Step Tracking
        self.step = 0

        # Render
        if renderEnv:
            pygame.init()
            self.canvas = pygame.display.set_mode((1000, 1000))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Env Simulation")

    def next(self, action):

        actionValue = [0.001, 0.01, 0, -0.01, -0.001][action]

        self.Satellite.run(self.Planet, self.G, actionValue)

        velocity = self.Satellite.velocity
        position = self.Satellite.position
        altitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)

        fuel = self.Satellite.fuel

        state = [velocity[0], velocity[1],
                 position[0], position[1], altitude, self.altitudeTarget, fuel]

        rewardAmount, done = self.reward()

        # Update self.rewardValue
        self.rewardValue += rewardAmount

        self.step += 1
        # done = False
        if (self.renderEnv):
            self.render(done)

        return (state, rewardAmount, done)

    def reward(self):

        rewardAmount = 0
        done = False

        altitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)

        # Check if the altitide is within 1 percent of the target
        if (altitude < self.altitudeTarget * 1.01 and altitude > self.altitudeTarget * 0.99):
            rewardAmount += 2

        # Check the fuel value compared to the previous fuel value
        if (self.Satellite.fuel < self.previousFuel):
            rewardAmount += -1

        # Calculate collisions
        if (utils.magnitude(self.Satellite.position - self.Planet.position) < self.Planet.size):
            rewardAmount += -100
            done = True

        # Calculate if the satellite has escaped
        if (utils.magnitude(self.Satellite.position - self.Planet.position) > 500):
            # rewardAmount += -100
            done = True

        # Calculate if the satellite has run out of fuel
        if (self.Satellite.fuel <= 0):
            rewardAmount += -100

        if (self.step >= self.runtime):
            done = True

        # Update the previous values
        self.previousFuel = self.Satellite.fuel

        # Return the reward and done values
        return (rewardAmount, done)

    # Render

    def render(self, done):
        if (self.render and not done):
            self.canvas.fill((255, 255, 255))
            self.Planet.render(self.canvas, 1)
            self.Satellite.render(self.canvas, 1)

            # Draw a circle with a radius of altitudeTarget and the origin at the center of the planet
            pygame.draw.circle(self.canvas, (0, 0, 0),
                               self.Planet.position, int(self.altitudeTarget), 1)

            # Display significant values
            font = pygame.font.Font('freesansbold.ttf', 12)
            text = font.render("Altitude: " + str(round(utils.magnitude(
                self.Satellite.position - self.Planet.position), 2)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (500, 50)
            self.canvas.blit(text, textRect)

            text = font.render(
                "Fuel: " + str(round(self.Satellite.fuel, 2)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (500, 70)
            self.canvas.blit(text, textRect)

            text = font.render(
                "Reward: " + str(round(self.rewardValue, 2)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (500, 90)
            self.canvas.blit(text, textRect)

            text = font.render("Step: " + str(self.step), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (500, 110)
            self.canvas.blit(text, textRect)

            # Update the display
            pygame.display.update()

    def reset(self):

        # Randomise the altitude target
        self.altitudeTarget = np.random.uniform(self.Planet.size + 10, 450)

        # Get a random position and velocity for the satellite within bounds
        velocityBounds = [-1, 1]

        # vy is always zero
        velocity = [np.random.uniform(velocityBounds[0], 0), np.random.uniform(
            velocityBounds[0], velocityBounds[1])]

        # Position is a random vector from the center of the planet
        position = utils.normalise(
            np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]))

        # Scalled randomly to be bigger than the planet but less than the screen size
        position = position * (self.Planet.size + (np.random.uniform(10, 490)))

        # Add the positions to make the planet the origin
        position = position + self.Planet.position

        self.Satellite.position = np.array(position)
        self.Satellite.velocity = np.array(velocity)

        # Reset the step counter and reward value
        self.step = 0
        self.rewardValue = 0

        # Set the fuel value
        Fuel = 50
        self.Satellite.fuel = Fuel

        # Calculate the initial altitude
        altitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)

        return [velocity[0], velocity[1], position[0], position[1], altitude, self.altitudeTarget, Fuel]

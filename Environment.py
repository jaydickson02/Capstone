import utils
import pygame
import numpy as np
from gym import spaces
import math


class Environment:

    # Constructor
    def __init__(self, dotAgent, Planet, Satellite, G, runtime, renderEnv=False):
        self.Planet = Planet
        self.Satellite = Satellite

        self.dotAgent = dotAgent

        self.G = G
        self.renderEnv = renderEnv
        self.runtime = runtime
        self.altitudeTarget = 0
        self.rewardValue = 0

        # Dot Agent
        self.targetArea1 = np.array([-50, -50]) + np.array([500, 500])
        self.targetArea2 = np.array([50, 50]) + np.array([500, 500])

        # Space Variables
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf, 0, 0, 0, 0, 0, 0, -np.inf, -np.inf, 0, 0, 0, 0, 0, 0]),
                                            high=np.array(
                                                [np.inf, np.inf, 1000, 1000, 1000, 1000, 1000, 1000, np.inf, np.inf, 1000, 1000, 1000, 1000, 1000, 1000]),
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

    def nextOld(self, action, previousState):
        actionValue = ['up', 'down', 'stay', 'left', 'right'][action]

        self.Satellite.run(self.Planet, self.G, actionValue)

        velocity = self.Satellite.velocity
        position = self.Satellite.position
        altitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)

        fuel = self.Satellite.fuel

        state = [velocity[0], velocity[1], position[0], position[1], altitude, self.altitudeTarget, fuel, previousState[0][0],
                 previousState[0][1], previousState[0][2], previousState[0][3], previousState[0][4], previousState[0][5], previousState[0][6]]

        rewardAmount, done = self.reward()

        # Update self.rewardValue
        self.rewardValue += rewardAmount

        self.step += 1
        # done = False
        if (self.renderEnv):
            self.render(done)

        return (state, rewardAmount, done)

    def next(self, action, previousState):
        actionValue = ['up', 'down', 'stay', 'left', 'right'][action]

        self.dotAgent.run(actionValue)

        velocity = self.dotAgent.velocity
        position = self.dotAgent.position

        state = [velocity[0], velocity[1], position[0], position[1], self.targetArea1[0], self.targetArea1[1], self.targetArea2[0], self.targetArea2[1],
                 previousState[0][0], previousState[0][1], previousState[0][2], previousState[0][3], previousState[0][4], previousState[0][5], previousState[0][6], previousState[0][7]]

        rewardAmount, done = self.reward()

        # Update self.rewardValue
        self.rewardValue += rewardAmount

        self.step += 1

        if (self.renderEnv):
            self.render(done)

        return (state, rewardAmount, done)

    def oldreward(self):

        rewardAmount = 0
        done = False

        altitude = utils.magnitude(
            self.Satellite.position - self.Planet.position)

        # Give reward depending on the closeness to the target altitude
        rewardAmount += 1/(abs(altitude - self.altitudeTarget))

        # Calculate collisions
        if (utils.magnitude(self.Satellite.position - self.Planet.position) < self.Planet.size):
            rewardAmount += -100
            done = True

        # Calculate if the satellite has escaped
        if (utils.magnitude(self.Satellite.position - self.Planet.position) > 500):
            # rewardAmount += -100
            done = True

        if (self.step >= self.runtime):
            done = True

        # Update the previous values
        self.previousFuel = self.Satellite.fuel

        # Return the reward and done values
        return (rewardAmount, done)

    def exponential_drop_off(self, x, target, A, k):
        return A * math.exp(-k * abs(x - target))

    def reward(self):

        rewardAmount = 0
        done = False

        # Calculate distance between target areas
        distance = utils.magnitude(self.targetArea1 - self.targetArea2)

        # Calculate the distance between the satellite and the target areas
        distance1 = utils.magnitude(self.dotAgent.position - self.targetArea1)
        distance2 = utils.magnitude(self.dotAgent.position - self.targetArea2)

        # Total distance between the target areas and the satellite
        DotDistanceTotal = distance1 + distance2

        # Calculate the reward
        rewardAmount += self.exponential_drop_off(
            DotDistanceTotal, distance, 1, 0.1)

        # Calculate if the satellite has escaped
        if (utils.magnitude(self.dotAgent.position - np.array([500, 500])) > 500):
            done = True

        if (self.step >= self.runtime):
            done = True

        # Return the reward and done values
        return (rewardAmount, done)

    # Render

    def oldrender(self, done):
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

    def render(self, done):
        if (self.render and not done):
            self.canvas.fill((255, 255, 255))
            self.dotAgent.render(self.canvas, 1)

            # Calculate the distance between the satellite and the target areas
            distance1 = utils.magnitude(
                self.dotAgent.position - self.targetArea1)
            distance2 = utils.magnitude(
                self.dotAgent.position - self.targetArea2)

            # Total distance between the target areas and the satellite
            DotDistanceTotal = distance1 + distance2

            # Render the targets
            pygame.draw.circle(self.canvas, [0, 0, 255], (int(
                self.targetArea1[0]), int(self.targetArea1[1])), 5)

            pygame.draw.circle(self.canvas, [255, 0, 0], (int(
                self.targetArea2[0]), int(self.targetArea2[1])), 5)

            # Display significant values
            font = pygame.font.Font('freesansbold.ttf', 12)
            text = font.render("Target Distance: " + str(round(utils.magnitude(
                self.targetArea1 - self.targetArea2), 2)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (500, 50)
            self.canvas.blit(text, textRect)

            text = font.render(
                "Current Distance: " + str(round(DotDistanceTotal, 2)), True, (0, 0, 0))
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

    def oldreset(self):

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

        return [velocity[0], velocity[1], position[0], position[1], altitude, self.altitudeTarget, Fuel, 0, 0, 0, 0, 0, 0, 0]

    def reset(self):

        # Randomise the target areas
        # self.targetArea1 = np.array(
        #     [np.random.uniform(-450, 450), np.random.uniform(-450, 450)])
        # self.targetArea2 = np.array(
        #     [np.random.uniform(-450, 450), np.random.uniform(-450, 450)])

        # Get a random position and velocity for the satellite within bounds
        velocityBounds = [-1, 1]

        # vy is always zero
        # velocity = [np.random.uniform(velocityBounds[0], 0), np.random.uniform(
        #     velocityBounds[0], velocityBounds[1])]

        velocity = np.array([0, 0])

        # Position is a random vector from the center of the planet
        position = utils.normalise(
            np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]))

        # Scalled randomly to be bigger than the planet but less than the screen size
        position = position * (np.random.uniform(10, 490))

        # Define a vector at the center of the screen
        center = np.array([500, 500])

        # Add the positions to make the planet the origin
        position = position + center

        self.dotAgent.position = np.array(position)
        self.dotAgent.velocity = np.array(velocity)

        # Reset the step counter and reward value
        self.step = 0
        self.rewardValue = 0

        return [velocity[0], velocity[1], position[0], position[1], self.targetArea1[0], self.targetArea1[1], self.targetArea2[0], self.targetArea2[1], 0, 0, 0, 0, 0, 0, 0, 0]

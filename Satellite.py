import numpy as np
import utils
import math
import pygame


class Satellite:

    # Constructor
    def __init__(self, position, velocity, size, colour=[0, 0, 0]):
        self.position = np.array([position[0], position[1]])
        self.velocity = np.array([velocity[0], velocity[1]])
        self.size = size
        self.colour = colour
        self.trailPoints = []
        self.fuel = 100

    # Method to run the satellite

    def run(self, Planet, G, tangentalAcceleration):

        gAcc = self.gravityAcceleration(Planet, G)
        acceleration = self.applyAcceleration(
            Planet, gAcc, tangentalAcceleration, "prograde")

        self.fuel = self.fuel - tangentalAcceleration

        self.propogate(acceleration)

    # Method to calculate the gravity acceleration
    def gravityAcceleration(self, Planet, G):

        # Calculate Gravity Vector
        gravityVector = Planet.position - self.position

        distanceVector = Planet.position - self.position

        # Normalize gravity vector
        gravityVector = utils.normalise(gravityVector)

        # Associated mag with 1/r^2 relationship
        gravityMag = (1/(utils.magnitude(distanceVector)**2)) * G

        gravityVector = gravityVector * gravityMag

        return gravityVector

    # Method to apply acceleration
    def applyAcceleration(self, Planet, gravityAcceleration, tangentalAcceleration, direction):

        distanceVector = Planet.position - self.position

        distanceVector = utils.normalise(distanceVector)

        # Get tangent vectors
        prograde = np.array([distanceVector[1], -distanceVector[0]])
        retrograde = np.array([distanceVector[0], distanceVector[1]])

        # Set Magnitudes
        prograde = prograde * tangentalAcceleration
        retrograde = retrograde * tangentalAcceleration

        # Set final Vector
        if direction == "prograde":
            # Add Accelerations
            acceleration = prograde + gravityAcceleration
            return (acceleration)

        elif direction == "retrograde":
            # Add Accelerations
            acceleration = retrograde + gravityAcceleration
            return acceleration

    # Method to update the position of the satellite
    def propogate(self, accelerationVector):

        # Update the position
        self.velocity = self.velocity + accelerationVector
        self.position = self.position + self.velocity

    # Find the starting velocity for a circular orbit

    def findOrbit(self, Planet, G):

        distanceVector = Planet.position - self.position

        gravity = (1/(utils.magnitude(distanceVector)**2)) * G

        initialV = math.sqrt(gravity * utils.magnitude(distanceVector))

        velocity = np.array([initialV, 0])

        self.velocity = velocity

    def render(self, screen, scale):

        pygame.draw.circle(screen, self.colour, (int(
            self.position[0] * scale), int(self.position[1] * scale)), self.size)

    def renderTrail(self, screen):

        trailLength = 500
        self.trailPoints.append((int(self.position[0]), int(self.position[1])))
        for i in range(0, len(self.trailPoints), 2):
            if i + 1 < len(self.trailPoints):
                pygame.draw.line(
                    screen, self.colour, self.trailPoints[i], self.trailPoints[i + 1], 1)
        if len(self.trailPoints) > trailLength:
            self.trailPoints.pop(0)

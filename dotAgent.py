import numpy as np
import utils
import math
import pygame


class dotAgent:

    # Constructor
    def __init__(self, position, velocity, size, colour=[0, 0, 0]):
        self.position = np.array([position[0], position[1]])
        self.velocity = np.array([velocity[0], velocity[1]])
        self.size = size
        self.colour = colour

    # Method to run the agent
    def run(self, direction):

        velocity = np.array([0, 0])

        if (direction == 'stay'):
            velocity = np.array([0, 0])
        elif (direction == 'up'):
            velocity = np.array([0, -1])
        elif (direction == 'down'):
            velocity = np.array([0, 1])
        elif (direction == 'left'):
            velocity = np.array([-1, 0])
        elif (direction == 'right'):
            velocity = np.array([1, 0])

        # Apply acceleration
        # self.velocity = self.velocity + acceleration
        self.position = self.position + velocity

    def render(self, screen, scale):

        pygame.draw.circle(screen, self.colour, (int(
            self.position[0] * scale), int(self.position[1] * scale)), self.size)

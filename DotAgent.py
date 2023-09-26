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
        self.previous_position = np.array([position[0], position[1]])

    # Method to run the agent
    def run(self, direction):

        self.velocity = np.array(direction)

        # Set Previous Position
        self.previous_position = self.position

        self.position = self.position + self.velocity

    def render(self, screen, scale):

        pygame.draw.circle(screen, self.colour, (int(
            self.position[0] * scale), int(self.position[1] * scale)), self.size)

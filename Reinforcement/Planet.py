import numpy as np
import pygame


class Planet:

    # Constructor
    def __init__(self, position, size, mass, color):
        self.position = np.array([position[0], position[1]])
        self.size = size
        self.mass = mass
        self.color = color

    # Method to render the planet

    def render(self, screen, scale):
        pygame.draw.circle(screen, self.color, (int(
            self.position[0] * scale), int(self.position[1] * scale)), self.size)

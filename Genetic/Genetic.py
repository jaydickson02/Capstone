from Planet import Planet
from Satellite import Satellite
import utils
import numpy as np
import pygame
import neat 



# Environment Parameters
G = 100
Width = 1000
Height = 1000  # Careful these are hardcoded in the environment for the reward function

 # Create objects
planet = Planet([Width/2, Height/2], 20, 1, [0, 0, 0])

sats = []

for x in range (1, 15):

    sat = Satellite([Width/2, Height/2 + (x * 20)], [0, 0], 3, [0, 0, 0])
    sat.findOrbit(planet, G)
    sats.append(sat)


pygame.init()
canvas = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()
pygame.display.set_caption("Environment Simulation for Genetic Test")

# # Render a circle

# pygame.draw.circle(canvas, (255, 0, 0), (500, 500), 20, 0)

actionValue = [0.001, 0.01, 0, -0.01, -0.001][2]

canvas.fill((255, 255, 255))
planet.render(canvas, 1)


while True:
    canvas.fill((255, 255, 255))
    planet.render(canvas, 1)

    for sat in sats:
        sat.run(planet, G, actionValue)
        sat.render(canvas, 1)
        sat.renderTrail(canvas)

    pygame.display.update()

# Random sample of the population
    # Randomly weighted neural network

# Parent explore the environment
    # Control an agent
    # There will be a task to perform

# Evluating the fitness of the population

# Selecting the best parents

# Crossover

# Mutation

# Repeat
import utils
import numpy as np
from gym import spaces
import math
import pygame


class Environment:

    # Constructor
    def __init__(self, dotAgent, runtime, renderEnv=False):

        self.dotAgent = dotAgent

        self.renderEnv = renderEnv
        self.runtime = runtime
        self.rewardValue = 0

        # Dot Agent
        self.targetArea = np.array([-50, -50]) + np.array([500, 500])
        self.obstacle = np.array([50, 50]) + np.array([500, 500])

        # Space Variables
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=np.array([-2, -2, -1000, -1000, -1000, -1000]),
                                            high=np.array(
                                                [2, 2, 1000, 1000, 1000, 1000]),
                                            dtype=np.float32)

        # Step Tracking
        self.step = 0

        # Render
        if renderEnv:
            pygame.init()
            self.canvas = pygame.display.set_mode((1000, 1000))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Env Simulation")

    def next(self, action, previousState):
        actionValue = [[0, 0], [-2, 0], [-1, 0], [1, 0], [2, 0], [0, -2],
                       [0, -1], [0, 1], [0, 2]][action]

        self.dotAgent.run(actionValue)

        relative_position_target = self.dotAgent.position - self.targetArea
        relative_position_obstacle = self.dotAgent.position - self.obstacle
        relative_velocity_target = self.dotAgent.velocity  # Assuming target is stationary

        state = [relative_velocity_target[0], relative_velocity_target[1], relative_position_target[0],
                 relative_position_target[1], relative_position_obstacle[0], relative_position_obstacle[1]]

        rewardAmount, done = self.reward()

        # Update self.rewardValue
        self.rewardValue += rewardAmount

        self.step += 1

        if (self.renderEnv):
            self.render(done)

        return (state, rewardAmount, done)

    def reward(self):

        rewardAmount = 0
        done = False

        # Define the parameters for the reward function
        def rho(s): return utils.magnitude(
            s - self.targetArea)  # Relative position of the agent to the target area

        # Relative position of the agent to the obstacle
        def rho_O(s): return utils.magnitude(s - self.obstacle)

        d_safe = 5  # Example safe distance
        gamma = 0.95  # Example discount factor

        # Calculate the reward using the paper's reward function
        s_prime = self.dotAgent.position
        s = self.dotAgent.previous_position  # Store the previous position

        if rho_O(s_prime) > 2 * d_safe:
            rewardAmount = -gamma * rho(s_prime) + rho(s)
        elif rho_O(s_prime) <= 2 * d_safe:
            rewardAmount = -rho_O(s) + 2 * (1 - gamma) * d_safe
        else:
            rewardAmount = -gamma * rho(s_prime) + \
                rho(s) + gamma * rho_O(s_prime)

        # Calculate if the satellite has escaped
        if (utils.magnitude(self.dotAgent.position - np.array([500, 500])) > 500):
            done = True

        if (self.step >= self.runtime):
            done = True

        # Return the reward and done values
        return (rewardAmount, done)

    # Render
    def render(self, done):
        if (self.render and not done):
            self.canvas.fill((255, 255, 255))
            self.dotAgent.render(self.canvas, 1)

            # Calculate the distance between the satellite and the target areas
            distance1 = utils.magnitude(
                self.dotAgent.position - self.targetArea)
            distance2 = utils.magnitude(
                self.dotAgent.position - self.targetArea)

            # Total distance between the target areas and the satellite
            DotDistanceTotal = distance1 + distance2

            # Render the targets
            pygame.draw.circle(self.canvas, [0, 0, 255], (int(
                self.targetArea[0]), int(self.targetArea[1])), 5)

            pygame.draw.circle(self.canvas, [255, 0, 0], (int(
                self.obstacle[0]), int(self.obstacle[1])), 5)

            # Display significant values
            font = pygame.font.Font('freesansbold.ttf', 12)
            text = font.render("Target Distance: " + str(round(utils.magnitude(
                self.targetArea - self.dotAgent.position), 2)), True, (0, 0, 0))
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

    def reset(self):

        # Define a vector at the center of the screen
        center = np.array([500, 500])

        # Randomise the target areas
        position1 = utils.normalise(
            np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]))
        position2 = utils.normalise(
            np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]))

        # Randomly scaled to be somewhere on the screen
        position1 = position1 * (np.random.uniform(0, 490))
        position2 = position2 * (np.random.uniform(0, 490))

        # Add the positions to make the center of the screen the origin
        position1 = position1 + center
        position2 = position2 + center

        self.targetArea = np.array(position1)
        self.obstacle = np.array(position2)

        # Get a random position and velocity for the satellite within bounds
        velocityBounds = [-1, 1]

        velocity = np.array([0, 0])  # Zero velocity for now

        # Position is a random direction from the center of the screen
        position = utils.normalise(
            np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]))

        # Randomly scaled to be somewhere on the screen
        position = position * (np.random.uniform(0, 490))

        # Add the positions to make the center of the screen the origin
        position = position + center

        self.dotAgent.position = np.array(position)
        self.dotAgent.velocity = np.array(velocity)

        relative_position_target = self.dotAgent.position - self.targetArea
        relative_position_obstacle = self.dotAgent.position - self.obstacle
        relative_velocity_target = self.dotAgent.velocity  # Assuming target is stationary

        # Reset the step counter and reward value
        self.step = 0
        self.rewardValue = 0

        return [relative_velocity_target[0], relative_velocity_target[1], relative_position_target[0],
                relative_position_target[1], relative_position_obstacle[0], relative_position_obstacle[1]]

import numpy as np
import utils
from gym import spaces
import math
import pygame


class Environment:

    # Constructor
    def __init__(self, dotAgent, runtime, renderEnv=False):

        # Environment Variables
        self.safe_distance = 5
        self.threat_distance = 10
        self.size = 300
        self.gamma = 0.95
        self.target_reached_range = 5


        # Dot Agent initialisation
        self.dotAgent = dotAgent([self.size/2, (self.size/2)], [0, 0], 3, [0, 0, 0])

        # Render Environment
        self.renderEnv = renderEnv

        # Max step value
        self.runtime = runtime

        # Initialise reward value
        self.rewardValue = 0

        # Initialise target area and obstacle
        self.targetArea = np.array([-50, -50]) + np.array([self.size/2, self.size/2])
        self.obstacle = np.array([50, 50]) + np.array([self.size/2, self.size/2])

        # Space Variables
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=np.array([-2, -2, -self.size, -self.size, -self.size, -self.size]),
        high=np.array([2, 2, self.size, self.size, self.size, self.size]), dtype=np.float32)

        # Step Tracking
        self.step = 0

        # Render
        if renderEnv:
            pygame.init()
            self.canvas = pygame.display.set_mode((self.size, self.size))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Env Simulation")

    def next(self, action):
        actionValue = [[0, 0], [-2, 0], [-1, 0], [1, 0], [2, 0], [0, -2],
                       [0, -1], [0, 1], [0, 2]][action]

        # Propogate the agent
        self.dotAgent.run(actionValue)

        # Calculate state values
        relative_position_target = self.dotAgent.position - self.targetArea
        relative_position_obstacle = self.dotAgent.position - self.obstacle
        relative_velocity_target = self.dotAgent.velocity  # Assuming target is stationary

        # Define the state
        state = [relative_velocity_target[0], relative_velocity_target[1], relative_position_target[0],
                 relative_position_target[1], relative_position_obstacle[0], relative_position_obstacle[1]]

        # Calculate the reward and check if done
        rewardAmount, done, finishCondition = self.reward()

        # Update self.rewardValue
        self.rewardValue += rewardAmount

        # Update the step counter
        self.step += 1

        if (self.renderEnv):
            self.render(done)

        return (state, rewardAmount, done, finishCondition)

    def reward(self):

        rewardAmount = 0
        done = False
        finishCondition = "No Condition"

        # Define the parameters for the reward function
        def rho(s): return utils.magnitude(
            s - self.targetArea)  # Relative position of the agent to the target area

        # Relative position of the agent to the obstacle
        def rho_O(s): return utils.magnitude(s - self.obstacle)

        d_safe = self.safe_distance  # safe distance
        gamma = self.gamma  # discount factor

        # Calculate the reward using the paper's reward function
        s_prime = self.dotAgent.position
        s = self.dotAgent.previous_position  # Store the previous position

        if rho_O(s_prime) > 2 * d_safe:
            rewardAmount += -gamma * rho(s_prime) + rho(s)

        if rho_O(s_prime) <= 2 * d_safe:
            rewardAmount += -rho_O(s) + 2 * (1 - gamma) * d_safe
        
        rewardAmount += -gamma * rho(s_prime) + rho(s) + gamma * rho_O(s_prime)

        # Calculate if the satellite collides with the obstacle
        if (utils.magnitude(self.dotAgent.position - self.obstacle) < self.threat_distance): 
            rewardAmount += -2  # Penalty for collision
            done = True
            finishCondition = "Collision"

        # Calculate if the satellite has gone out of range
        if any(abs(self.dotAgent.position - (self.size/2)) > (self.size/2)): 
            rewardAmount += -2  # Penalty for going out of range
            done = True
            finishCondition = "Out of Range"

        # Calculate if the satellite has reached the target area
        if (utils.magnitude(self.dotAgent.position - self.targetArea) < self.target_reached_range): 
            rewardAmount += 2  # Reward for reaching the target
            done = True
            finishCondition = "Reached Target"

        if (self.step >= self.runtime):
            done = True
            finishCondition = "Runtime Exceeded"

        # Return the reward and done values
        return (rewardAmount, done, finishCondition)

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
            textRect.center = (150, 50)
            self.canvas.blit(text, textRect)

            text = font.render(
                "Current Distance: " + str(round(DotDistanceTotal, 2)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (150, 70)
            self.canvas.blit(text, textRect)

            text = font.render(
                "Reward: " + str(round(self.rewardValue, 2)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (150, 90)
            self.canvas.blit(text, textRect)

            text = font.render("Step: " + str(self.step), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (150, 110)
            self.canvas.blit(text, textRect)

            # Update the display
            pygame.display.update()

    def reset(self):

        # Define a vector at the center of the screen
        center = np.array([150, 150])

        # Randomise the target areas
        position1 = utils.normalise(
            np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]))
        position2 = utils.normalise(
            np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]))

        # Randomly scaled to be somewhere on the screen
        position1 = position1 * (np.random.uniform(0, self.size/2))
        position2 = position2 * (np.random.uniform(0, self.size/2))

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
        position = position * (np.random.uniform(0, self.size/2))

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

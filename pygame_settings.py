import random
import operator
import numpy as np
import pygame
import sys
import collections
import os
import pathlib


# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
ORANGE = (255, 128, 0)

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 60
HEIGHT = 60

# This sets the margin between each cell
MARGIN = 1
wall, clear, goal = "#", ".", "*"
grid = ["...*....",
        "...#..*.",
        "........",
        "........",
        "........",
        "........",
        "#####.##",
        "........"]

class Game:

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    A = [UP, DOWN, LEFT, RIGHT, STAY]
    A_DIFF = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]


    def __init__(self):

        self.num_agents = 2
        self.bump = 0
        self.step_bump = 0
        self.num_landmarks = 2
        self.state_size = (self.num_agents + self.num_landmarks) * 2
        self.grid_size = 8
        self.render_flag = True
        self.counter_step = 0
        # enables visualizer
        if self.render_flag:
            [self.screen, self.my_font] = self.gui_setup()
            self.step_num = 1


        self.cells = []
        self.positions_idx = []

        # self.agents_collide_flag = args['collide_flag']
        # self.penalty_per_collision = args['penalty_collision']
        self.num_episodes = 0
        self.terminal = False
        img = pygame.image.load('goal.jpeg').convert()
        self.img_goal = pygame.transform.scale(img, (WIDTH, WIDTH))
        img = pygame.image.load('robot.jpg').convert()
        self.img_robot = pygame.transform.scale(img, (WIDTH, WIDTH))
        img = pygame.image.load('robot_on_target.jpeg').convert()
        self.img_robot_on_target = pygame.transform.scale(img, (WIDTH, WIDTH))


    def gui_setup(self):
        pygame.init()
        board_size_x = (WIDTH + MARGIN) * self.grid_size
        board_size_y = (HEIGHT + MARGIN) * self.grid_size

        window_size_x = int(board_size_x * 1.01)
        window_size_y = int(board_size_y * 1.2)

        window_size = [window_size_x, window_size_y]
        screen = pygame.display.set_mode(window_size, 0, 32)

        # Set title of screen
        pygame.display.set_caption("MultiAgentRL game")

        myfont = pygame.font.SysFont("bold", 30)

        return [screen, myfont]

    def render(self):

        pygame.time.delay(500)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(BLACK)
        text = self.my_font.render("Step: {0}".format(self.step_num), 1, WHITE)
        self.screen.blit(text, (5, 15))

        for row in range(self.grid_size):
            for column in range(self.grid_size):
                pos = (row, column)

                frequency = self.find_frequency(pos, self.agents_positions)

                if pos in self.landmarks_positions and frequency >= 1:
                    if frequency == 1:
                        self.screen.blit(self.img_robot_on_target,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                    
                elif pos in self.landmarks_positions:
                    self.screen.blit(self.img_goal,
                                     ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                elif frequency >= 1:
                    if frequency == 1:
                        self.screen.blit(self.img_robot,
                                         ((MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50))
                else:
                    pygame.draw.rect(self.screen, WHITE,[(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN + 50, WIDTH,HEIGHT])

        for i in range(8):
            if i != 5 :
                pygame.draw.rect(self.screen, BLACK, [(MARGIN + WIDTH) * i + MARGIN, (MARGIN + HEIGHT) * 6 + MARGIN + 50, WIDTH, HEIGHT])

        pygame.draw.rect(self.screen, BLACK,
                         [(MARGIN + WIDTH) * 3 + MARGIN, (MARGIN + HEIGHT) * 1 + MARGIN + 50, WIDTH, HEIGHT])


        pygame.display.update()

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]

        #positions_idx = [3, 14, 17, 31, 40, 47, 56, 63]

        positions_idx = [3, 14, 56, 63]

        return [cells, positions_idx]


    def reset(self):

        self.bump = 0
        self.terminal = False
        [self.cells, self.positions_idx] = self.set_positions_idx()

        # separate the generated position indices for walls, pursuers, and evaders
        landmarks_positions_idx = self.positions_idx[0:self.num_landmarks]
        agents_positions_idx = self.positions_idx[self.num_landmarks:self.num_landmarks + self.num_agents]

        # map generated position indices to positions
        self.landmarks_positions = [self.cells[pos] for pos in landmarks_positions_idx]
        self.agents_positions = [self.cells[pos] for pos in agents_positions_idx]

        initial_state = list(sum(self.landmarks_positions + self.agents_positions, ()))

        return initial_state

    def bfs(self, start):
        queue = collections.deque([[start]])
        seen = set([start])
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if grid[y][x] == goal:
                return path
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= x2 < self.grid_size and 0 <= y2 < self.grid_size and grid[y2][x2] != wall and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))


    def update_positions(self, pos_list, act_list):
        positions_action_applied = []
        self.step_bump = 0
        for idx in range(len(pos_list)):
            if act_list[idx] != 4:
                pos_act_applied = list(map(operator.add, pos_list[idx], self.A_DIFF[act_list[idx]]))
                # checks to make sure the new pos in inside the grid
                for i in range(0, 2):
                    if pos_act_applied[i] < 0:
                        pos_act_applied[i] = 0
                        self.step_bump += 1
                    if pos_act_applied[i] >= self.grid_size:
                        pos_act_applied[i] = self.grid_size - 1
                        self.step_bump += 1

                    if pos_act_applied[0] == 6:
                        if pos_act_applied[1] != 5:
                            pos_act_applied = list(pos_list[idx])
                            self.step_bump += 1
                    if pos_act_applied[0] == 1:
                        if pos_act_applied[1] == 3:
                            pos_act_applied = list(pos_list[idx])
                            self.step_bump += 1
                positions_action_applied.append(tuple(pos_act_applied))
            else:
                positions_action_applied.append(pos_list[idx])

        final_positions = []

        for pos_idx in range(len(pos_list)):
            if positions_action_applied[pos_idx] == pos_list[pos_idx]:
                final_positions.append(pos_list[pos_idx])
            elif positions_action_applied[pos_idx] not in pos_list and positions_action_applied[
                pos_idx] not in positions_action_applied[
                                0:pos_idx] + positions_action_applied[
                                             pos_idx + 1:]:
                final_positions.append(positions_action_applied[pos_idx])
            else:
                final_positions.append(pos_list[pos_idx])
                self.step_bump += 1

        self.bump += self.step_bump
        return final_positions

    def step(self, agents_actions):
        self.counter_step += 1
        idx = 1
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)
        binary_cover_list = []

        for landmark in self.landmarks_positions:
            distances = [np.linalg.norm(np.array(landmark) - np.array(agent_pos), 1)
                         for agent_pos in self.agents_positions]



            min_dist = min(distances)
            if min_dist == 0:
                binary_cover_list.append(0)
            else:
                binary_cover_list.append(1)

            #idx-=1

        reward = -1 * sum(binary_cover_list)

        # check the terminal case
        if reward == 0:
            self.terminal = True
            reward += 200
        else:
            self.terminal = False
        if self.counter_step >= 900000:
            if self.step_bump != 0:
                reward -= 1

        #reward -= self.step_bump
        new_state = list(sum(self.landmarks_positions + self.agents_positions, ()))
        self.step_num += 1

        return [new_state, reward, self.terminal]


#    def print_bump(self):
#        print(self.bump)

    def record_step(self, episode, step):

        current_path = os.getcwd()
        snaps_path = os.path.join(current_path, 'results')
        snaps_path = os.path.join(snaps_path, 'snaps')
        pygame.image.save(self.screen, snaps_path + "/screenshot_{}_step_{}.jpeg".format(episode, step))

    def find_frequency(self, a, items):
        freq = 0
        for item in items:
            if item == a:
                freq += 1

        return freq


    def action_space(self):
        return len(self.A)

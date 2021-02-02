
import numpy as np
import os
import random
import argparse
import pandas as pd
from agent import Agent
from pygame_settings import Game

class Setup(object):

    def __init__(self):
        self.game = Game()
        self.episodes_number = 5000
        self.max_ts = 200
        self.filling_steps = 0
        self.steps_b_updates = 4
        self.test = False
        self.episodes = []
        self.rewards = []
        self.bumps = []

    def run(self, agents, file1, file2):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        max_score = -10000
        for episode_num in range(self.episodes_number):
            state = self.game.reset()
            self.game.render()


            # converting list of positions to an array
            state = np.array(state)
            state = state.ravel()

            done = False
            reward_all = 0
            time_step = 0
            while not done and time_step < self.max_ts:

                actions = []
                for agent in agents:
                    actions.append(agent.greedy_actor(state))
                next_state, reward, done = self.game.step(actions)


                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                if not self.test:
                    for agent in agents:
                        agent.observe((state, actions, reward, next_state, done))
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            if time_step % self.steps_b_updates == 0:
                                agent.replay()
                            agent.update_target_model()

                total_step += 1
                time_step += 1
                state = next_state
                reward_all += reward

                self.game.render()
                if episode_num >= 2500:
                    self.game.record_step(episode_num, time_step)

            rewards_list.append(reward_all)
            timesteps_list.append(time_step)

            print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}, Bumps: {f}".format(p=episode_num, s=reward_all,
                                                                               t=time_step, g=done, f=self.game.bump))

            self.episodes.append(episode_num)
            self.bumps.append(self.game.bump)
            self.rewards.append(reward_all)


            if not self.test:
                for agent in agents:
                    if episode_num ==1500:
                        agent.memory_model = "UER"
                        agent.memory = agent.memory2
                if episode_num % 2 == 0:
                    df = pd.DataFrame(rewards_list, columns=['score'])
                    df.to_csv(file1)

                    df = pd.DataFrame(timesteps_list, columns=['steps'])
                    df.to_csv(file2)

                    if total_step >= self.filling_steps:
                        if reward_all > max_score:
                            for agent in agents:
                                agent.brain.save_model()
                            max_score = reward_all



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # DQN Parameters
    parser.add_argument('-l', '--learning-rate', default=0.00005, type=float, help='Learning rate')
    parser.add_argument('-op', '--optimizer', choices=['Adam', 'RMSProp'], default='Adam',
                        help='Optimization method')
    parser.add_argument('-m', '--memory-capacity', default=10000, type=int, help='Memory capacity')
    parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('-t', '--target-frequency', default=2000, type=int,
                        help='Number of steps between the updates of target network')
    parser.add_argument('-x', '--maximum-exploration', default=100000, type=int, help='Maximum exploration step')
    parser.add_argument('-nn', '--number-nodes', default=256, type=int, help='Number of nodes in each layer of NN')
    parser.add_argument('-mt', '--memory', choices=['UER', 'PER'], default='PER')
    parser.add_argument('-pl', '--prioritization-scale', default=0.5, type=float, help='Scale for prioritization')
    parser.add_argument('-test', '--test', action='store_true', help='Enable the test phase if "store_false"')
    parser.add_argument('-gn', '--gpu-num', default='0', type=str, help='Number of GPU to use')


    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_num']
    env = Setup()
    game = Game()

    state_size = env.game.state_size
    action_space = env.game.action_space()

    all_agents = []
    for b_idx in range(env.game.num_agents):

        brain_file = './results/weights_files/' + '_weights_' + str(b_idx) + '.h5'
        all_agents.append(Agent(state_size, action_space, b_idx, brain_file, args))

    rewards_file ='./results/rewards_files/' + '_rewards.csv'
    timesteps_file ='./results/timesteps_files/' + '_timestep.csv'
    env.run(all_agents, rewards_file, timesteps_file)
    print(env.episodes)
    print(env.rewards)
    print(env.bumps)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

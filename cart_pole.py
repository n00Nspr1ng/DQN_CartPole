import numpy as np
import torch
from torch import nn, optim
import gym
import matplotlib.pyplot as plt

from DQN_model import agent, experience_replay
from q_net import Q_net


class CartPole():
    def __init__(self, num_episodes, max_steps, show_graph=None):
        '''
        Class for wrapping gym CartPole-v0
        @ num_episodes : the number of episodes DQN will train on
        @ max_steps : the max number of step in one episode
        @ show_graph : whether to show the graph of total rewards and Q value after the end of training
        '''
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.show_graph = show_graph


    def init_agent(self, env):
        ''' Initialize Q network and create agent to train '''

        num_state = env.observation_space.shape[0]
        num_action = env.action_space.n

        learning_rate = 0.001
        batch_size = 16
        gamma = 0.98
        memory_capacity = 1000

        model = Q_net(num_state, num_action)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        replay_memory = experience_replay.experienceReplay(memory_capacity)

        return agent.Agent(num_action, gamma, replay_memory, batch_size, model, criterion, optimizer)


    def run(self):
        ''' Running the CartPole agent '''

        best_list = []
        history = {'Q_val':[], 'total_reward':[]}

        self.env = gym.make('CartPole-v0')
        self.agent = self.init_agent(self.env)

        for episode in range(self.num_episodes):
            
            total_reward = 0
            state = self.env.reset()
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)
            done = False

            for step in range(self.max_steps):
                if done:
                    best_list.append(step)
                    history["Q_val"].append(self.agent.get_q(state, action).detach().numpy()[0][0])
                    history["total_reward"].append(total_reward)
                    break

                action = self.agent.get_action(state, episode)
                next_state, reward, done, _ = self.env.step(action.item())
                done_mask = torch.FloatTensor([0.0]) if done else torch.FloatTensor([1.0])
                
                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                next_state = torch.unsqueeze(next_state, 0)

                total_reward += reward
                reward = torch.FloatTensor([reward])

                self.agent.save(state, action, next_state, reward, done_mask)
                self.agent.train()

                if next_state != None:
                    state = next_state

            if done:
                # 최근 100 episode 평균 reward
                len_best_list = len(best_list[-100:])
                avg_best_list = sum(best_list[-100:]) / len_best_list

                print('episode {:4.0f}: \tstep: {:4.0f} \tscore: {:4f}, \taverage step({:3f}): {:6.2f}'
                    .format(episode, step+1, total_reward, len_best_list, avg_best_list ) )

        self.env.env.close()

        if self.show_graph:

            fig1 = plt.figure(figsize=(12,5))
            ax1 = fig1.add_subplot(121)
            #print(shape(history["Q_val"]))
            ax1.plot(history['total_reward'], color='blue', linewidth=0.8)
            ax1.set_title("Total Reward")
            ax1.set_xlabel("Episode")
            
            ax2 = fig1.add_subplot(122)
            ax2.plot(history["Q_val"], color='red', linewidth=0.8)
            ax2.set_title("Q value")
            ax2.set_xlabel("Episode")

            plt.show()


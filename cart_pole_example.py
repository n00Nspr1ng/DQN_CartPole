import numpy as np
import torch
from torch import nn, optim
import gym

from DQN_model import agent, experience_replay


class Q_net(nn.Module):
    def __init__(self, num_state, num_action):
        super().__init__()
    
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, num_action)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x



class CartPole():
    def __init__(self, num_episodes, max_steps):
        self.num_episodes = num_episodes
        self.max_steps = max_steps


    def init_agent(self, env):
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

        best_list = []

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
                    break
                
                action = self.agent.get_action(state, episode)
                next_state, reward, done, _ = self.env.step(action.item())
                done_mask = torch.FloatTensor([0.0]) if done else torch.FloatTensor([1.0])
                
                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                next_state = torch.unsqueeze(next_state, 0)

                total_reward += reward
                reward = torch.FloatTensor([reward])

                self.agent.save(state, action, next_state, reward , done_mask)
                self.agent.train()

                state = next_state

            if done:
                # 최근 100 episode 평균 reward
                len_best_list = len(best_list[-100:])
                avg_best_list = sum(best_list[-100:]) / len_best_list

                print('episode {:4.0f}: \tstep: {:4.0f} \tscore: {:4f}, \taverage step({:3f}): {:6.2f}'
                    .format(episode, step+1, total_reward, len_best_list, avg_best_list ) )

        self.env.env.close()




if __name__ == "__main__":
    cart_pole = CartPole(num_episodes=500, max_steps=1000)
    cart_pole.run()
    

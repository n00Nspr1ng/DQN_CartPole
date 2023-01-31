import torch
import random

from . import experience_replay, util

class Agent:
    def __init__(self, num_action, gamma, replay_memory, batch_size, model, criterion, optimizer):
        '''
        Class for agent of DQN model
        @ num_action : the number of possible actions in the environment's action space
        @ gamma : discout factor
        @ replay_memory : the buffer of replay memory
        @ batch_size : the number of batch size sampled from replay buffer that the model will randomly train on
        @ model : the Q net
        @ criterion : criterion for the Q net
        @ optimizer : optimizer for the Q net
        '''
        
        self.num_action = num_action
        self.gamma = gamma
        self.replay_memory = replay_memory

        self.batch_size = batch_size
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.last_q_value = None


    def train(self):
        '''Function for updating Q value'''
        
        ##### Wait until the memory is full #####
        if len(self.replay_memory) < self.batch_size:
            return

        ##### Change type that the net can train on #####
        sampled_memory = self.replay_memory.sample(self.batch_size)
        batch = util.Transition(*zip(*sampled_memory))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_mask_batch = torch.cat(batch.done_mask)
        
        ##### Find current Q & target Q #####
        self.model.eval()
        # Find Q_values according to the current action
        Q_value = self.model(state_batch).gather(1, action_batch)
        self.last_q_value = Q_value

        next_Q_value = torch.zeros(self.batch_size)
        max_next_Q_value = self.model(next_state_batch).max(1)[0].detach()
        Q_target = (reward_batch + self.gamma * max_next_Q_value * done_mask_batch).unsqueeze(1)

        ##### Train the Q net #####
        self.model.train()

        loss = self.criterion(Q_value, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_action(self, state, episode):
        ''' Get action with ε-greedy '''

        ##### Define the probability epsilon #####
        epsilon = 1 / (2*(episode + 1))

        ##### Get action using ε-greedy policy ε-greedy
        if epsilon <= random.uniform(0,1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).argmax(1).view(1,1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_action)]])

        return action
    

    def save(self, state, action, next_state, reward, done_mask):
        ''' Save the tuple in the replay buffer '''

        return self.replay_memory.save(state, action, next_state, reward, done_mask)


    def get_q(self, state, action):
        ''' Get the value of Q with the given state and action '''

        return self.model(state).gather(1, action)

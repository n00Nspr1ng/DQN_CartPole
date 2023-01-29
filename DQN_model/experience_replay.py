import random

from . import common

class experienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay_memory = []
        self.index = 0
    
    
    def save(self, state, action, next_state, reward, done_mask):
        ''' Save (s, a, s', r, done_mask) as a tuple into buffer dictionary '''
        transition = common.Transition(state, action, next_state, reward, done_mask)

        if len(self.replay_memory) < self.capacity:
            self.replay_memory.append(transition)
        else:
            self.replay_memory[self.index] = transition
        
        self.index = (self.index + 1) % self.capacity


    def sample(self, batch_size):
        ''' Sample from the replay memory with batch size '''
        return random.sample(self.replay_memory, batch_size)

    def __len__(self):
        ''' Call self using 'len' function '''
        return len(self.replay_memory)







if __name__=="__main__":
    print("start")

    replay = experienceReplay(3)
    replay.save(1, 3, 5, 7, 0)
    replay.save(2, 4, 6, 8, 0)
    replay.save(3, 5, 7, 9, 0)
    replay.save(4, 6, 8, 10, 1)

    print(replay.sample(2))
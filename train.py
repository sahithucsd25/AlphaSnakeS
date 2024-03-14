import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import *
import gym
import gym_snake
import math
import numpy as np

# Assume the following classes and functions are defined:
# QNetwork - a PyTorch Module that represents the Q-network
# preprocess - a function that preprocesses raw images to the desired input format
# Environment - a class that provides methods `reset()` and `step(action)` to interact with the game    

def preprocess(rgb_grid):
    greyscale_grid = np.dot(rgb_grid[...,:3], [0.299, 0.587, 0.114])
    return greyscale_grid

# Hyperparameters
N = 10000          # Capacity of the replay memory
BATCH_SIZE = 32    # Size of the minibatch
GAMMA = 0.99       # Discount factor
EPS_START = 1.0    # Starting value of epsilon
EPS_END = 0.1      # Minimum value of epsilon
EPS_DECAY = 200    # Rate at which epsilon should decay
TARGET_UPDATE = 40 # Update the target network every fixed number of steps



def train():
    # Initialize replay memory
    replay_memory = deque(maxlen=N)

    # Initialize action-value function Q with random weights
    q_network = DQN()
    target_network = DQN()
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.AdamW(
        q_network.parameters(), lr=1e-4, eps=1e-8
        )
    
    # Implement epsilon-greedy policy
    def select_action(state, epsilon):
        # print(state.shape)
        if random.random() > epsilon:
            with torch.no_grad():
                return q_network(state.float()).argmax(1).view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], dtype=torch.long)
    
    # Setup the environment
    env = gym.make('snake-v0', grid_size=(24, 24))
    # env.grid_size = (24, 24)
    state = env.reset()
    # print(state.shape)
    # state = preprocess(state)

    # Training loop
    num_episodes = 1000
    steps_done = 0
    for episode in range(num_episodes):
        state = env.reset()
        prev_state = preprocess(state)
        first_action = random.randrange(1, 4)
        curr_state, _, _, _ = env.step(first_action)
        curr_state = preprocess(curr_state) # 24x24
        state = torch.tensor(np.stack((curr_state, prev_state)), dtype=torch.float).unsqueeze(0) # 1x2x24x24
        # print(state.shape)
        
        while True: # for t in count():
            # Select and perform an action
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            action = select_action(state, epsilon)
            curr_state, reward, done, _ = env.step(action.item())
            curr_state = preprocess(curr_state)
            next_state = torch.concat((torch.tensor((curr_state), dtype=torch.float).unsqueeze(0), state[0][0].unsqueeze(0))).unsqueeze(0) # 1x2x24x24
            
            # Store the transition in memory
            replay_memory.append((state, action, reward, next_state, done))
            
            # Move to the next state
            state = next_state
            
            # Perform one step of the optimization
            if len(replay_memory) > BATCH_SIZE:
                transitions = random.sample(replay_memory, BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                
                # Compute a mask of non-final states
                non_final_mask = torch.tensor(tuple(map(lambda t: t == False, batch_done)), dtype=torch.bool)
                non_final_next_states = torch.cat([s for s, t in zip(batch_next_state, batch_done) if not t])
                
                # Compute Q values
                # print(type(batch_state))
                state_action_values = q_network(torch.cat(batch_state)).gather(1, torch.tensor(batch_action).unsqueeze(1))
                next_state_values = torch.zeros(BATCH_SIZE)
                next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()
                
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + torch.tensor(batch_reward, dtype=torch.float)
                
                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in q_network.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
            
            if done:
                break

            steps_done += 1
        
            # Update the target network every fixed number of steps
            if steps_done % TARGET_UPDATE == 0:
                target_network.load_state_dict(q_network.state_dict())

        if episode % 50 == 0:
            torch.save(q_network.state_dict(), f"network.pth")
            print(f"Episode {episode+1} complete")

import time
import torch
import torch.nn.functional as F
import numpy as np
import gym
import gym_snake
import random
from collections import deque
import matplotlib.pyplot as plt
from torch.optim import Adam
from copy import deepcopy
from models2 import DQN

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

def update_policy(net, target_net, optimizer, memory, batch_size, gamma):
    if len(memory) < batch_size:
        return
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    current_q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    loss = F.mse_loss(current_q_values, expected_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    env = gym.make('snake-v0', n_foods=9) 
    n_actions = env.action_space.n

    policy_net = DQN(in_channels=3, n_actions=n_actions)
    target_net = deepcopy(policy_net)
    for param in target_net.parameters():
        param.requires_grad = False

    optimizer = Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayBuffer(10000)
    batch_size = 32
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    episodes = 1000

    steps_done = 0
    for episode in range(episodes):
        state = env.reset()
        state = np.transpose(state, (2, 0, 1)) 
        total_reward = 0
        done = False
        while not done:
            # if episode % 1 == 0:  # render
            #         env.render()
            #         time.sleep(0.05)

            epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                      np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension and adjust channel
                    action = policy_net(state_tensor).max(1)[1].view(1, 1).item()

            next_state, reward, done, _ = env.step(action)
            next_state = np.transpose(next_state, (2, 0, 1))

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            update_policy(policy_net, target_net, optimizer, memory, batch_size, gamma)
        env.close()
        # plt.close()

        if episode % 10 == 0:  # Update the target network every 10 episodes
            target_net.load_state_dict(policy_net.state_dict())
        
        print(f'Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.2f}')

    env.close()

if __name__ == "__main__":
    set_seed(42)
    main()
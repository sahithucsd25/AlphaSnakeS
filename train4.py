import torch
import torch.nn.functional as F
import numpy as np
import gym
import gym_snake
import random
import time
from collections import deque
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
    
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def main():
    initial_foods = 5  # Start with more foods
    min_foods = 1  # Minimum number of foods
    food_decay_rate = 5  # Number of episodes to reduce the number of foods
    env = gym.make('snake-v0', n_foods=initial_foods, grid_size=(6,6))
    n_actions = env.action_space.n

    policy_net = DQN(in_channels=6, n_actions=n_actions)
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
    episodes = 100000

    fruit_100 = []
    steps_100 = []

    steps_done = 0
    for episode in range(episodes):
        current_foods = max(min_foods, initial_foods - episode // food_decay_rate)
        #current_foods = 1
        env = gym.make('snake-v0', n_foods=current_foods, grid_size=(6,6))
        state = env.reset()
        state = np.transpose(state, (2, 0, 1))
        prev_state = np.zeros_like(state)  # Initialize prev state as zeros
        
        total_steps = 0
        fruits_eaten = 0
        total_reward = 0
        done = False
        while not done:
            total_steps += 1
            if episode % 1000 == 0:  # render
                env.render()
                time.sleep(0.5)

            combined_state = np.concatenate((prev_state, state), axis=0)  # Combine current and previous states
            
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                      np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    combined_state_tensor = torch.FloatTensor(combined_state).unsqueeze(0)
                    action = policy_net(combined_state_tensor).max(1)[1].view(1, 1).item()

            next_state, reward, done, _ = env.step(action)

            #print(reward)

            if (reward == 2):
                fruits_eaten += 1

            next_state = np.transpose(next_state, (2, 0, 1))
            next_combined_state = np.concatenate((state, next_state), axis=0)  # Next combined state for memory
            
            memory.push(combined_state, action, reward, next_combined_state, done)
            
            state = next_state  # Update the current state
            total_reward += reward
            prev_state = state.copy()  # Update the previous state

            update_policy(policy_net, target_net, optimizer, memory, batch_size, gamma)
        env.close()
        

        if episode % 10 == 0:  # Update the target network
            target_net.load_state_dict(policy_net.state_dict())

        fruit_100.append(fruits_eaten)
        steps_100.append(total_steps)

        if episode % 100 == 0:
            print(f"Episode: {episode}, 100 Fruit Average: {np.mean(fruit_100)}, 100 Step Average: {np.mean(steps_100)}")
            fruit_100 = []
            steps_100 = []
        
        #print(f'Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon:.2f}, Fruits eaten: {fruits_eaten}, Steps: {total_steps}')

    env.close()

if __name__ == "__main__":
    set_seed(42)
    main()

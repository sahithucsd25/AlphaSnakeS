import sys
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gym
import gym_snake
import numpy as np
import time
import matplotlib.pyplot as plt
from model import DQN  # Ensure this import points to your DQN architecture


def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess(rgb_grid):
    return np.dot(rgb_grid[..., :3], [0.299, 0.587, 0.114])

def get_priority_sampling_weights(replay_memory):
    # Extract rewards and compute the absolute values
    rewards = np.exp(np.array([memory[2].cpu() for memory in replay_memory]))
    # Compute a simple priority measure - higher reward gets higher probability
    priorities = rewards / np.sum(rewards)
    return priorities

def prepare_batch(batch_state, batch_action, batch_reward, batch_next_state, batch_done):
    # Stack for PyTorch compatibility
    batch_state = torch.stack(batch_state)  # Should result in [32, 2, 150, 150] if each state is [2, 150, 150]
    batch_next_state = torch.stack(batch_next_state)
    batch_action = torch.tensor(batch_action, dtype=torch.long).view(-1, 1)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float)
    batch_done = torch.tensor(batch_done, dtype=torch.float)

    return batch_state, batch_action, batch_reward, batch_next_state, batch_done


def sample_from_replay_memory(replay_memory, batch_size):
        # Get sampling probabilities
        sampling_probs = get_priority_sampling_weights(replay_memory).flatten()
        # Sample indices according to the probabilities
        batch_indices = np.random.choice(
            range(len(replay_memory)), size=batch_size, p=sampling_probs
        )
        # Retrieve the memories
        sampled_memories = [replay_memory[idx] for idx in batch_indices]

        return zip(*sampled_memories)

def random_sample_from_replay_memory(replay_memory, batch_size):
    return zip(*random.sample(replay_memory, batch_size))


class Trainer:
    def __init__(self, grid_size="10x10", episodes=10000, batch_size=32, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.999, target_update=2, replay_memory_size=10000):
        self.grid_size = grid_size
        self.dim = int(grid_size[:2])
        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = gym.make("snake-v0", grid_size=(int(grid_size[:2]), int(grid_size[:2])), n_foods=1)
        self.q_network = DQN(self.dim, self.device)
        self.target_network = DQN(self.dim, self.device)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=1e-3, eps=1e-8)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.steps_done = 0
        self.epsilon = self.eps_start
        

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.q_network(state.float()).argmax(1).view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], dtype=torch.long, device=self.device)

    def optimize_model(self):
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = random_sample_from_replay_memory(self.replay_memory, self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = transitions

        batch_state_tensor = torch.cat([s.unsqueeze(0) for s in batch_state], dim=0)
        batch_next_state_tensor = torch.cat([s.unsqueeze(0) for s in batch_next_state], dim=0)
        stacked_states = torch.cat((batch_state_tensor, batch_next_state_tensor), dim=1).to(self.device)


        non_final_mask = torch.tensor(tuple(map(lambda s: s is not True, batch_done)), dtype=torch.bool, device=self.device)
        non_final_next_states = stacked_states[non_final_mask]

        state_action_values = self.q_network(stacked_states).gather(1, torch.tensor(batch_action, device=self.device).unsqueeze(1))
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + torch.tensor(batch_reward, device=self.device)

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for episode in range(self.episodes):
            if (episode % 20 == 0):
                self.env = gym.make("snake-v0", grid_size=(10, 10), n_foods=30)


            current_state = preprocess(self.env.reset())
            current_state = torch.tensor([current_state], device=self.device, dtype=torch.float)
            prev_state = torch.zeros_like(current_state)

            ep_reward = 0
            steps_this_episode = 0
            while True:
                if episode % 100 == 0:  # render
                    self.env.render()
                    time.sleep(0.05)

                state = torch.cat((prev_state, current_state), dim=0) 
                action = self.select_action(state)
                next_state_raw, reward, done, _ = self.env.step(action.item())
                next_state_raw = preprocess(next_state_raw)
                next_state = torch.tensor([next_state_raw], device=self.device, dtype=torch.float)

                ep_reward += reward
                reward = torch.tensor([reward], device=self.device, dtype=torch.float)

                self.replay_memory.append((current_state, action, reward, next_state, done))
                prev_state = current_state
                current_state = next_state

                self.optimize_model()
                self.steps_done += 1
                steps_this_episode += 1

                if done:
                    break

                if self.steps_done % self.target_update == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.env.close()
            plt.close()

            self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
            if episode % 100 == 0 or episode == self.episodes - 1:  # Also print for the last episode
                mean_reward = ep_reward / steps_this_episode if steps_this_episode else 0
                avg_steps_per_episode = self.steps_done / (episode + 1)
                print(f'Episode {episode}, Mean reward: {mean_reward:.2f}, Average steps per episode: {avg_steps_per_episode:.2f}, Epsilon: {self.epsilon}')

        print('Training complete')


if __name__ == "__main__":
    set_seed(9) 
    trainer = Trainer()
    trainer.train()
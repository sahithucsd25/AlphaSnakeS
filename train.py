import sys
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import *
import gym
import gym_snake
import math
import time
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # Numpy module.
    random.seed(seed_value)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(9)


def preprocess(rgb_grid):
    greyscale_grid = np.dot(rgb_grid[..., :3], [0.299, 0.587, 0.114])
    return greyscale_grid


# Hyperparameters
N = 10000  # Capacity of the replay memory
BATCH_SIZE = 32  # Size of the minibatch
GAMMA = 0.99  # Discount factor
EPISODES = 10000
EPS_START = 1.0  # Starting value of epsilon
EPS_END = 0.05  # Minimum value of epsilon
EPS_DECAY = 0.999  # Rate at which epsilon should decay
TARGET_UPDATE = 2  # Update the target network every fixed number of steps


def train():
    replay_memory = deque(maxlen=N)
    device = 0 if len(sys.argv) == 1 else int(sys.argv[1])
    grid_size = "15x15"
    width = int(grid_size[:2])
    q_network = DQN(width, device)
    target_network = DQN(width, device)
    device = q_network.device
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.AdamW(q_network.parameters(), lr=1e-3, eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    def select_action(state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                return q_network(state.float()).argmax(1).view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(4)]], dtype=torch.long, device=device
            )

    def get_priority_sampling_weights(replay_memory):
        # Extract rewards and compute the absolute values
        rewards = np.exp(np.array([memory[2] for memory in replay_memory]))
        # Compute a simple priority measure - higher reward gets higher probability
        priorities = rewards / np.sum(rewards)
        return priorities

    def sample_from_replay_memory(replay_memory, batch_size):
        # Get sampling probabilities
        sampling_probs = get_priority_sampling_weights(replay_memory)
        # Sample indices according to the probabilities
        batch_indices = np.random.choice(
            range(len(replay_memory)), size=batch_size, p=sampling_probs
        )
        # Retrieve the memories
        sampled_memories = [replay_memory[idx] for idx in batch_indices]
        return zip(*sampled_memories)

    num_episodes = EPISODES
    steps_done = 0
    prev_steps = 0
    reward_per_5 = 0
    steps_per_5 = 0
    initial_foods = 30  # Start with 10 foods
    food_decrease_episodes = 1000
    file_name = f"network{grid_size}_same_1.pth"

    epsilon = EPS_START
    for episode in range(num_episodes):
        current_foods = max(3, initial_foods)
        if episode % 10 == 0:  # increase this value?
            env = gym.make(
                "snake-v0", grid_size=(width, width), n_foods=current_foods
            )  # , random_init=False

        ep_reward = 0
        state = env.reset()
        prev_state = preprocess(state)
        first_action = int(random.randrange(0, 4))
        curr_state, _, _, _ = env.step(first_action)
        curr_state = preprocess(curr_state)  # 24x24
        state = (
            torch.tensor(np.stack((curr_state, prev_state)), dtype=torch.float)
            .unsqueeze(0)
            .to(device)
        )
        # epsilon *= EPS_DECAY
        # epsilon = max(EPS_END, epsilon)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        while True:  # for t in count():
            # Select and perform an action
            # if episode % 500 == 0:  # render
            #     env.render()
            #     time.sleep(0.05)

            action = select_action(state, epsilon)
            curr_state, reward, done, _ = env.step(action.item())
            ep_reward += reward
            curr_state = preprocess(curr_state)
            next_state = (
                torch.concat(
                    (
                        torch.tensor(
                            (curr_state), dtype=torch.float, device=device
                        ).unsqueeze(0),
                        state[0][0].unsqueeze(0),
                    )
                )
                .unsqueeze(0)
                .to(device)
            )

            replay_memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay_memory) > BATCH_SIZE:
                transitions = sample_from_replay_memory(replay_memory, BATCH_SIZE)
                (
                    batch_state,
                    batch_action,
                    batch_reward,
                    batch_next_state,
                    batch_done,
                ) = transitions

                # Compute a mask of non-final states
                non_final_mask = torch.tensor(
                    tuple(map(lambda t: t == False, batch_done)), dtype=torch.bool
                )
                print("bns", len(batch_next_state), len(batch_next_state[0]), len(batch_next_state[0][0]), len(batch_next_state[0][0][0]))
                non_final_next_states = torch.cat(
                    [s for s, t in zip(batch_next_state, batch_done) if not t]
                )
                print(non_final_next_states.shape)

                # Compute Q values
                # print(len(batch_state), len(batch_state[0]), len(batch_state[0][0]), len(batch_state[0][0][0]))
                # print(len(batch_action[0]))

                # print(torch.cat(batch_state).shape)
                state_action_values = q_network(torch.cat(batch_state)).gather(
                    1, torch.tensor((batch_action), device=device).unsqueeze(1)
                )
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                next_state_values[non_final_mask] = (
                    target_network(non_final_next_states).max(1)[0].detach()
                )
                print(len(next_state_values[non_final_mask]))
                print(next_state_values.shape)

                # Compute the expected Q values
                expected_state_action_values = (
                    next_state_values * GAMMA
                ) + torch.tensor(batch_reward, dtype=torch.float, device=device)

                # Compute Huber loss
                loss = F.smooth_l1_loss(
                    state_action_values, expected_state_action_values.unsqueeze(1)
                )

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # for param in q_network.parameters():
                #     param.grad.data.clamp_(-1, 1)
                optimizer.step()
                # scheduler.step()

            if done:
                break

            steps_done += 1

            # Update the target network every fixed number of steps
            if steps_done % TARGET_UPDATE == 0:
                target_network.load_state_dict(q_network.state_dict())
        env.close()
        plt.close()

        if (episode + 1) % 500 != 0:
            reward_per_5 += ep_reward
            steps_per_5 += steps_done - prev_steps
        if (episode + 1) % 500 == 0:
            torch.save(q_network.state_dict(), file_name)
            print(
                f"Episode: {episode+1:5d}, Mean Reward: {reward_per_5/500:.2f}, Average steps taken: {steps_per_5/500:.2f}"
            )
            reward_per_5 = 0
            steps_per_5 = 0

        prev_steps = steps_done

    # save model
    torch.save(q_network.state_dict(), f"network{grid_size}.pth")


if __name__ == "__main__":
    train()

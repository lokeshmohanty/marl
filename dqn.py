import numpy as np
import torch
from torch import nn
from collections import namedtuple, deque
from random import sample
import matplotlib.pyplot as plt
from rich.progress import track
from copy import deepcopy

def epsilon_greedy(q_values, epsilon=0.1):
    if torch.rand(1)[0] < epsilon:
        return torch.randint(0, q_values.shape[0], (1,)).item()
    else:
        return torch.argmax(q_values).item()

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 24)
        self.l2 = nn.Linear(24, 24)
        self.l3 = nn.Linear(24, action_dim)

    def forward(self, x):
        x = nn.functional.relu(self.l1(x))
        x = nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)

class DQN:
    def __init__(self, env, batch_size=128):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.env = env
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = deepcopy(self.critic)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.loss_fn = nn.HuberLoss()
        self.memory = ReplayBuffer(int(1e5))
        self.rewards = []
        self.means = []
        self.tau = 5e-3
        self.gamma = 0.99
        self.batch_size = batch_size

    def select_action(self, state):
        return epsilon_greedy(self.critic(torch.tensor(state, dtype=torch.float32)))

    def train(self, max_eps=500, max_iter=int(1e3)):
        for e in range(max_eps):
            state = self.env.reset()[0]
            episode_reward = 0
            for _ in track(range(max_iter),
                           description=f"Episode({e+1}/{max_eps})",
                           transient=True):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = 1 if terminated or truncated else 0
                self.memory.push(state, action, next_state, reward, done)
                state = next_state
                episode_reward += reward

                if len(self.memory) > self.batch_size:
                    batch = self.memory.sample(self.batch_size)
                    with torch.no_grad():
                        states, actions, next_states, rewards, dones = [
                            torch.from_numpy(np.array(x, dtype=np.float32)) for x in zip(*batch)
                        ]
                        actions = actions.to(dtype=torch.int64)

                    q_values = self.critic(states)
                    next_q_values = self.critic_target(next_states)
                    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                    next_q_values = next_q_values.max(1).values.detach()
                    target = rewards + self.gamma * next_q_values * (1 - dones)

                    with torch.no_grad():
                        for p, tp in zip(self.critic.parameters(),
                                        self.critic_target.parameters()):
                            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

                    loss = self.loss_fn(q_values, target)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if done:
                    self.rewards.append(episode_reward)
                    episode_reward = 0
                    self.plot()
                    break

        # self.plot(show_result=True)
        # plt.show()

    def plot(self, show_result=False):
        plt.figure(1)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(self.rewards)

        self.means.append(np.mean(self.rewards[-min(len(self.rewards),
                                                    self.batch_size):]))
        plt.plot(self.means)
        plt.pause(0.001)

import numpy as np
import torch
from torch import nn
from collections import namedtuple, deque
from random import sample
import matplotlib.pyplot as plt
from rich.progress import track
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = nn.functional.relu(self.l1(state))
        a = nn.functional.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        q1 = nn.functional.relu(self.l1(x))
        q1 = nn.functional.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = nn.functional.relu(self.l4(x))
        q2 = nn.functional.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        x = torch.cat([state, action], 1)
        q1 = nn.functional.relu(self.l1(x))
        q1 = nn.functional.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

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

class TD3:
    def __init__(self, env, batch_size=128):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.env = env
        self.actor = Actor(state_dim, action_dim, env.action_space.high[0])
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.actor_loss_fn = nn.HuberLoss()
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.critic_loss_fn = nn.HuberLoss()
        self.memory = ReplayBuffer(int(1e5))
        self.rewards = []
        self.means = []
        self.tau = 5e-3
        self.gamma = 0.99
        self.batch_size = batch_size

    def select_action(self, state):
        return self.actor(torch.tensor(state, dtype=torch.float32)).detach().numpy()
        
    def train(self, max_eps=1000, max_iter=int(1e3)):
        for e in range(max_eps):
            state = self.env.reset()[0]
            episode_reward = 0
            for i in track(range(max_iter),
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

                    with torch.no_grad():
                        next_actions = self.actor_target(next_states)
                        target_q1, target_q2 = self.critic_target(next_states, next_actions)
                        target_q = torch.min(target_q1, target_q2)
                        target = rewards + self.gamma * (1 - dones) * target_q

                    q1, q2 = self.critic(states, actions)
                    critic_loss = self.critic_loss_fn(q1, target) + self.critic_loss_fn(q2, target)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    if i % 4 == 0:
                        actor_loss = -self.critic.q1(states, self.actor(states)).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()


                        with torch.no_grad():
                            for p, tp in zip(self.critic.parameters(),
                                            self.critic_target.parameters()):
                                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
                            for p, tp in zip(self.actor.parameters(),
                                            self.actor_target.parameters()):
                                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

                if done:
                    self.rewards.append(episode_reward)
                    episode_reward = 0
                    self.plot()
                    break

        self.plot(show_result=True)
        plt.show()

    def plot(self, show_result=False):
        plt.figure(1)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.rewards)

        self.means.append(np.mean(self.rewards[-min(len(self.rewards),
                                                    self.batch_size):]))
        plt.plot(self.means)
        plt.pause(0.001)

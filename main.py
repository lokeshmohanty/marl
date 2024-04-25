import typer
from rich.progress import track
from enum import StrEnum
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn import DQN
from td3 import TD3

app = typer.Typer()

Environment = StrEnum("Environment", [
    "CartPole-v1",
    "LunarLander-v2",
    "LunarLanderContinuous-v2",
    "Humanoid-v5",
    "Hopper-v5",
    "Pursuit-v4",
])
Algorithm = StrEnum("Algorithm", ["DQN", "TD3", "SAC"])


@app.command()
def train(env: Environment, alg: Algorithm):
    print(f"Training\nAlgorithm: {alg.name}\nEnvironment: {env.name}")
    gymEnv = gym.make(env.name)
    runs = []
    match alg.name:
        case "DQN":
            for i in range(10):
                print(f"Run {i+1}/10 ---")
                dqn = DQN(gymEnv)
                train_data = dqn.train()
                runs.append(train_data)
        case "TD3":
            td3 = TD3(gymEnv)
            train_data = td3.train()
        case _:
            raise ValueError(f"Algorithm {alg.name} not implemented")

    np.save(f"train_data_{alg.name}_{env.name}.npy", runs)

@app.command()
def plot(env: Environment, alg: Algorithm):
    train_data = np.load(f"train_data_{alg.name}_{env.name}.npy",
                         allow_pickle=True)
    print(train_data)
    data = np.vstack(train_data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    plt.plot(mean, label="Mean")
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    plt.title(f"{alg} on {env}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

@app.command()
def test(env: Environment, alg: Algorithm, model: str):
    print(f"Testing: (alg: {alg.name}, env: {env.name})")
    gymEnv = gym.make(env.name, render_mode="human")
    max_duration = gymEnv.spec.max_episode_steps or int(1e3)
    obs = gymEnv.reset()[0]
    for e in range(10):
        for _ in track(range(max_duration), description=f"Episode({e+1}/10):"):
            action = gymEnv.action_space.sample()
            obs, reward, terminated, truncated, _ = gymEnv.step(action)
            if terminated or truncated:
                obs = gymEnv.reset()[0]
                break
    gymEnv.close()


@app.command()
def view(env: Environment):
    print(f"View: (env: {env.name})")
    print(f"Observation Space: {gym.make(env.name).observation_space}")
    print(f"Action Space: {gym.make(env.name).action_space}")

    if env.name == "Pursuit-v4":
        from pettingzoo.sisl import pursuit_v4

        gymEnv = pursuit_v4.parallel_env(render_mode="human")
        observations, infos = gymEnv.reset()

        while gymEnv.agents:
            # this is where you would insert your policy
            actions = {agent: gymEnv.action_space(agent).sample() for agent in gymEnv.agents}
            _, _, terminations, truncations, infos = gymEnv.step(actions)
        gymEnv.close()


    gymEnv = gym.make(env.name, render_mode="human")
    max_duration = gymEnv.spec.max_episode_steps or int(1e3)
    gymEnv.reset()
    for e in range(10):
        for _ in track(range(max_duration), description=f"Episode({e+1}/10):"):
            action = gymEnv.action_space.sample()
            _, _, terminated, truncated, _ = gymEnv.step(action)
            if terminated or truncated:
                gymEnv.reset()
                break
    gymEnv.close()

if __name__ == "__main__":
    app()

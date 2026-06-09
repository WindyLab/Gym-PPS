import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import gym

from agent import DDPGAgent
from arguments import get_args
from custom_env import MyAction, MyObs, MyReward, predator_prey_distance
from replay_buffer import ReplayBuffer


def make_env():
    custom_param = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_param.json")
    env = gym.make("PredatorPreySwarm-v0")
    env = MyReward(env, custom_param)
    env = MyAction(env, custom_param)
    env = MyObs(env, custom_param)
    assert env.n_p == 1 and env.n_e == 1, "This minimal DDPG script expects 1 predator and 1 prey."
    return env


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    obs = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DDPGAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
    )
    replay = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    print(f"1v1 PPS: obs_dim={obs_dim}, predator_action_dim={act_dim}, policy_action_shape={env.action_space.shape}")
    total_steps = 0
    best_min_distance = float("inf")

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        distance = predator_prey_distance(env)
        episode_reward = 0.0
        min_distance = distance
        caught_once = False
        noise = max(args.noise_min, args.noise * (1.0 - episode / args.episodes))
        render_episode = args.render or episode % 20 == 0

        for _ in range(args.episode_length):
            if render_episode:
                env.render(mode="human")

            if total_steps < args.start_steps:
                pred_action = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
            else:
                pred_action = agent.act(state, noise_std=noise)

            next_state, rewards, _, _ = env.step(pred_action)
            reward = float(rewards[0, 0])
            distance = predator_prey_distance(env)
            caught = distance <= env.size_p + env.size_e
            caught_once = caught_once or caught
            replay.push(state, pred_action, reward, next_state, False)

            state = next_state
            episode_reward += reward
            min_distance = min(min_distance, distance)
            total_steps += 1

            if len(replay) >= args.batch_size and total_steps >= args.start_steps:
                for _ in range(args.updates_per_step):
                    agent.update(replay, args.batch_size)

        best_min_distance = min(best_min_distance, min_distance)
        if episode % 10 == 0 or caught_once:
            print(
                f"episode={episode:04d} reward={episode_reward:8.3f} "
                f"min_dist={min_distance:.4f} best={best_min_distance:.4f} "
                f"caught={caught_once} noise={noise:.3f}"
            )

    torch.save(agent.actor.state_dict(), os.path.join(os.path.dirname(__file__), args.model_path))
    env.close()


if __name__ == "__main__":
    train(get_args())

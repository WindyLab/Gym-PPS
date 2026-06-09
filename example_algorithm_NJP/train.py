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
from custom_env import MyAction, MyObs, MyReward, predator_prey_distances
from replay_buffer import ReplayBuffer


def make_env():
    custom_param = os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_param.json")
    env = gym.make("PredatorPreySwarm-v0")
    env = MyReward(env, custom_param)
    env = MyAction(env, custom_param)
    env = MyObs(env, custom_param)
    assert env.n_p > 0 and env.n_e > 0, "This shared DDPG script expects at least 1 predator and 1 prey."
    return env


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    obs = env.reset()
    obs_dim = env.observation_space.spaces["predator"].shape[1]
    act_dim = env.action_space.spaces["predator"].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predator_agent = DDPGAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
    )
    prey_agent = DDPGAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        device=device,
    )
    predator_replay = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    prey_replay = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    print(
        f"NvsN PPS: n_p={env.n_p}, n_e={env.n_e}, obs_dim={obs_dim}, "
        f"shared_action_dim={act_dim}"
    )
    total_steps = 0
    best_min_distance = float("inf")

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        predator_states = obs["predator"]
        prey_states = obs["prey"]
        pred_distances = predator_prey_distances(env)
        episode_reward = 0.0
        episode_prey_reward = 0.0
        min_distance = float(np.min(pred_distances))
        caught_once = False
        noise = max(args.noise_min, args.noise * (1.0 - episode / args.episodes))
        render_episode = args.render or episode % 20 == 0

        for _ in range(args.episode_length):
            if render_episode:
                env.render(mode="human")

            predator_actions = np.zeros((act_dim, env.n_p), dtype=np.float32)
            for pred_i in range(env.n_p):
                if total_steps < args.start_steps:
                    predator_actions[:, pred_i] = np.random.uniform(-1.0, 1.0, size=act_dim)
                else:
                    predator_actions[:, pred_i] = predator_agent.act(predator_states[pred_i], noise_std=noise)

            prey_actions = np.zeros((act_dim, env.n_e), dtype=np.float32)
            for prey_i in range(env.n_e):
                if total_steps < args.start_steps:
                    prey_actions[:, prey_i] = np.random.uniform(-1.0, 1.0, size=act_dim)
                else:
                    prey_actions[:, prey_i] = prey_agent.act(prey_states[prey_i], noise_std=noise)

            action = {"predator": predator_actions, "prey": prey_actions}
            next_obs, rewards, _, _ = env.step(action)
            pred_rewards = rewards["predator"]
            prey_rewards = rewards["prey"]
            pred_distances = predator_prey_distances(env)
            caught = bool(np.min(pred_distances) <= env.size_p + env.size_e)
            next_predator_states = next_obs["predator"]
            next_prey_states = next_obs["prey"]
            caught_once = caught_once or caught
            for pred_i in range(env.n_p):
                predator_replay.push(
                    predator_states[pred_i],
                    predator_actions[:, pred_i],
                    pred_rewards[pred_i],
                    next_predator_states[pred_i],
                    False,
                )
            for prey_i in range(env.n_e):
                prey_replay.push(
                    prey_states[prey_i],
                    prey_actions[:, prey_i],
                    prey_rewards[prey_i],
                    next_prey_states[prey_i],
                    False,
                )

            predator_states = next_predator_states
            prey_states = next_prey_states
            episode_reward += float(np.mean(pred_rewards))
            episode_prey_reward += float(np.mean(prey_rewards))
            min_distance = min(min_distance, float(np.min(pred_distances)))
            total_steps += 1

            if total_steps >= args.start_steps:
                for _ in range(args.updates_per_step):
                    if len(predator_replay) >= args.batch_size:
                        predator_agent.update(predator_replay, args.batch_size)
                    if len(prey_replay) >= args.batch_size:
                        prey_agent.update(prey_replay, args.batch_size)

        best_min_distance = min(best_min_distance, min_distance)
        if episode % 10 == 0 or caught_once:
            print(
                f"episode={episode:04d} pred_reward={episode_reward:8.3f} "
                f"prey_reward={episode_prey_reward:8.3f} "
                f"min_dist={min_distance:.4f} best={best_min_distance:.4f} "
                f"caught={caught_once} noise={noise:.3f}"
            )

    torch.save(
        {
            "predator_actor": predator_agent.actor.state_dict(),
            "prey_actor": prey_agent.actor.state_dict(),
        },
        os.path.join(os.path.dirname(__file__), args.model_path),
    )
    env.close()


if __name__ == "__main__":
    train(get_args())

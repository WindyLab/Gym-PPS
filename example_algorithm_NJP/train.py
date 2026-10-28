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
    # 在创建环境前校验图执行依赖，避免错误配置启动训练后才失败。
    if args.cuda_graph and (not args.fused_adam or not torch.cuda.is_available()):
        raise ValueError("cuda_graph requires CUDA and --fused_adam")
    if args.render_interval < 0:
        raise ValueError("render_interval must be nonnegative")
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
        fused_adam=args.fused_adam,
        cuda_graph=args.cuda_graph,
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
        fused_adam=args.fused_adam,
        cuda_graph=args.cuda_graph,
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
        render_episode = args.render or (args.render_interval > 0 and episode % args.render_interval == 0)

        for _ in range(args.episode_length):
            if render_episode:
                env.render(mode="human")

            if total_steps < args.start_steps:
                # 按个体排列随机数，保持原逐个体采样顺序。
                predator_actions = np.random.uniform(-1.0, 1.0, size=(env.n_p, act_dim)).astype(np.float32).T
                prey_actions = np.random.uniform(-1.0, 1.0, size=(env.n_e, act_dim)).astype(np.float32).T
            else:
                # 每个物种只推理一次，再转成环境要求的 (动作维度, 个体数)。
                predator_actions = predator_agent.act_batch(predator_states, noise_std=noise).T
                prey_actions = prey_agent.act_batch(prey_states, noise_std=noise).T

            action = {"predator": predator_actions, "prey": prey_actions}
            next_obs, rewards, _, _ = env.step(action)
            pred_rewards = rewards["predator"]
            prey_rewards = rewards["prey"]
            pred_distances = predator_prey_distances(env)
            caught = bool(np.min(pred_distances) <= env.size_p + env.size_e)
            next_predator_states = next_obs["predator"]
            next_prey_states = next_obs["prey"]
            caught_once = caught_once or caught
            # 按原个体顺序整批写入，减少逐个体调用，抽样和终止标记保持原样。
            predator_replay.push_batch(predator_states, predator_actions.T,
                                       pred_rewards, next_predator_states)
            prey_replay.push_batch(prey_states, prey_actions.T,
                                  prey_rewards, next_prey_states)

            predator_states = next_predator_states
            prey_states = next_prey_states
            episode_reward += float(np.mean(pred_rewards))
            episode_prey_reward += float(np.mean(prey_rewards))
            min_distance = min(min_distance, float(np.min(pred_distances)))
            total_steps += 1

            if total_steps >= args.start_steps:
                for _ in range(args.updates_per_step):
                    if len(predator_replay) >= args.batch_size:
                        predator_agent.update(predator_replay, args.batch_size, return_losses=False)
                    if len(prey_replay) >= args.batch_size:
                        prey_agent.update(prey_replay, args.batch_size, return_losses=False)

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
    if args.cuda_graph:
        print(f"CUDA Graph: predator={predator_agent.cuda_graph_status}, "
              f"prey={prey_agent.cuda_graph_status}, "
              f"capture_seconds={predator_agent.graph_capture_seconds + prey_agent.graph_capture_seconds:.3f}")


if __name__ == "__main__":
    train(get_args())

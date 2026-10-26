import argparse
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent import DDPGAgent
from custom_env import predator_prey_distances
from train import (
    make_env,
)


def get_args():
    parser = argparse.ArgumentParser("Evaluate two shared DDPG policies for N-vs-N Gym-PPS")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="ddpg_both_NvsN.pt")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def evaluate(args):
    if args.episodes <= 0 or args.episode_length <= 0:
        raise ValueError("episodes and episode_length must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    obs = env.reset()
    obs_dim = env.observation_space.spaces["predator"].shape[1]
    act_dim = env.action_space.spaces["predator"].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predator_agent = DDPGAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim, device=device)
    prey_agent = DDPGAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim, device=device)
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
    # 评估只需要两个 Actor；完整检查点先放在 CPU，避免回放缓冲占用显存。
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    predator_agent.actor.load_state_dict(checkpoint["predator_actor"])
    prey_agent.actor.load_state_dict(checkpoint["prey_actor"])
    # 网络权重已复制到模型，立即释放不参与评估的训练状态。
    del checkpoint
    predator_agent.actor.eval()
    prey_agent.actor.eval()

    contact_distance = env.size_p + env.size_e
    successes = 0
    min_distances = []
    catch_steps = []

    print(
        f"Evaluating {os.path.basename(model_path)}: "
        f"n_p={env.n_p}, n_e={env.n_e}, obs_dim={obs_dim}, "
        f"shared_action_dim={act_dim}, contact_distance={contact_distance:.4f}"
    )

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        predator_states = obs["predator"]
        prey_states = obs["prey"]
        distances = predator_prey_distances(env)
        min_distance = float(np.min(distances))
        caught_once = False
        first_catch_step = args.episode_length

        for step in range(1, args.episode_length + 1):
            if args.render:
                env.render(mode="human")

            # 每个物种批量推理一次，再转成环境要求的 (动作维度, 个体数)。
            predator_actions = predator_agent.act_batch(predator_states).T
            prey_actions = prey_agent.act_batch(prey_states).T

            action = {"predator": predator_actions, "prey": prey_actions}
            next_obs, _, _, _ = env.step(action)
            predator_states = next_obs["predator"]
            prey_states = next_obs["prey"]
            distances = predator_prey_distances(env)
            distance = float(np.min(distances))
            min_distance = min(min_distance, distance)

            if distance <= contact_distance and not caught_once:
                caught_once = True
                first_catch_step = step

        successes += int(caught_once)
        min_distances.append(min_distance)
        catch_steps.append(first_catch_step)
        print(
            f"episode={episode:03d} caught={caught_once} "
            f"first_catch_step={first_catch_step:03d} min_dist={min_distance:.4f}"
        )

    success_rate = successes / args.episodes
    print(
        f"summary: success_rate={success_rate:.2%}, "
        f"avg_min_dist={np.mean(min_distances):.4f}, "
        f"avg_steps={np.mean(catch_steps):.1f}"
    )
    env.close()


if __name__ == "__main__":
    evaluate(get_args())

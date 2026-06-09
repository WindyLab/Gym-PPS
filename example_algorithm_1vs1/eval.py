import argparse
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent import DDPGAgent
from custom_env import predator_prey_distance
from train import make_env


def get_args():
    parser = argparse.ArgumentParser("Evaluate minimal 1v1 DDPG for Gym-PPS")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="ddpg_predator_1v1.pt")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def evaluate(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env()
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DDPGAgent(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim, device=device)
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
    try:
        actor_state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        actor_state = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()

    contact_distance = env.size_p + env.size_e
    successes = 0
    min_distances = []
    catch_steps = []

    print(
        f"Evaluating {os.path.basename(model_path)}: "
        f"obs_dim={obs_dim}, predator_action_dim={act_dim}, contact_distance={contact_distance:.4f}"
    )

    # for episode in range(1, args.episodes + 1):
    for episode in range(3):
        state = env.reset()
        min_distance = predator_prey_distance(env)
        caught_once = False
        first_catch_step = args.episode_length

        for step in range(1, args.episode_length + 1):
            if args.render:
                env.render(mode="human")

            pred_action = agent.act(state, noise_std=0.0)
            state, _, _, _ = env.step(pred_action)
            distance = predator_prey_distance(env)
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

import argparse


def get_args():
    parser = argparse.ArgumentParser("Two shared DDPG policies for N-vs-N Gym-PPS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--buffer_size", type=int, default=5e5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--start_steps", type=int, default=1000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--noise_min", type=float, default=0.05)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model_path", type=str, default="ddpg_both_NvsN.pt")
    return parser.parse_args()

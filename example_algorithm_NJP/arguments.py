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
    # 两个开关共用同一配置项，后出现者生效，方便覆盖启动脚本预设的 fused 后端。
    parser.add_argument("--fused_adam", action="store_true", help="Use fused Adam on CUDA; CPU keeps the default backend.")
    parser.add_argument("--no_fused_adam", dest="fused_adam", action="store_false", help="Use the default Adam backend, overriding --fused_adam.")
    parser.set_defaults(fused_adam=False)
    # 图执行必须显式开启；两个开关按出现顺序覆盖，默认仍使用普通更新。
    parser.add_argument("--cuda_graph", action="store_true",
                        help="Capture DDPG updates; requires CUDA and --fused_adam.")
    parser.add_argument("--no_cuda_graph", dest="cuda_graph", action="store_false")
    parser.set_defaults(cuda_graph=False)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--noise_min", type=float, default=0.05)
    parser.add_argument("--render", action="store_true")
    # 间隔为 0 时关闭周期渲染，便于无窗口训练。
    parser.add_argument(
        "--render_interval", type=int, default=20,
        help="Render every N episodes; 0 disables periodic rendering.",
    )
    parser.add_argument("--model_path", type=str, default="ddpg_both_NvsN.pt")
    return parser.parse_args()

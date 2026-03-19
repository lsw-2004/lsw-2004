import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from unity_env import UnityNavEnv, EnvConfig


# =========================
# Config
# =========================
@dataclass
class EvalConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 改成你的 Unity build 路径
    file_name: str = r"D:\DRL_Navigation\Builds\Project_1.exe"

    # 和 Unity 当前 behavior 保持一致
    behavior_name: str = "Navtest?team=0"

    # 改成你想评估的 checkpoint
    checkpoint_path: str = r"./checkpoints/reward_shaping_15.0/ppo_update_0400.pt"

    # 评估 episode 数
    num_eval_episodes: int = 50

    # 是否显示 Unity 窗口
    no_graphics: bool = False

    # 环境参数：要和训练/基线评估保持一致
    obs_dim: int = 187
    action_dim: int = 2
    lidar_dim: int = 180
    reach_goal_radius: float = 0.5
    max_steps: int = 200

    # reward 参数：建议与训练时一致，保证 return 可比
    progress_gain: float = 1.5
    time_penalty: float = -0.002
    collision_penalty: float = -2.0
    success_bonus: float = 15.0
    timeout_penalty: float = -2.0
    near_obstacle_threshold: float = 0.4
    near_obstacle_penalty: float = -0.05
    action_l2_penalty: float = -0.0005


# =========================
# Model (需与 train_ppo.py 保持一致)
# =========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.backbone(x)
        mean = self.actor_mean(h)
        value = self.critic(h)
        logstd = self.actor_logstd.expand_as(mean)
        return mean, logstd, value

    def get_deterministic_action(self, x):
        mean, _, value = self(x)
        return mean, value.squeeze(-1)


# =========================
# Evaluation
# =========================
def evaluate_one_episode(env: UnityNavEnv, model: ActorCritic, device: torch.device):
    obs_np, info = env.reset()
    done = False

    ep_ret = 0.0
    ep_len = 0
    last_info = info

    while not done:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_mean, _ = model.get_deterministic_action(obs)

        action_np = action_mean.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)

        obs_np, reward, done, truncated, info = env.step(action_np)

        ep_ret += reward
        ep_len += 1
        last_info = info

    result = {
        "return": float(ep_ret),
        "length": int(ep_len),
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "timeout": bool(last_info.get("timeout", False)),
        "final_goal_dist": float(last_info.get("goal_dist", np.nan)),
    }
    return result


def summarize_results(results):
    success_rate = np.mean([r["success"] for r in results])
    collision_rate = np.mean([r["collision"] for r in results])
    timeout_rate = np.mean([r["timeout"] for r in results])
    avg_len = np.mean([r["length"] for r in results])
    avg_ret = np.mean([r["return"] for r in results])
    avg_final_goal_dist = np.mean([r["final_goal_dist"] for r in results])

    print("\n=== PPO Evaluation Summary ===")
    print(f"success_rate        : {success_rate:.3f}")
    print(f"collision_rate      : {collision_rate:.3f}")
    print(f"timeout_rate        : {timeout_rate:.3f}")
    print(f"avg_len             : {avg_len:.2f}")
    print(f"avg_return          : {avg_ret:.3f}")
    print(f"avg_final_goal_dist : {avg_final_goal_dist:.3f}")


def main():
    cfg = EvalConfig()
    device = torch.device(cfg.device)

    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"找不到 checkpoint: {cfg.checkpoint_path}")

    env_cfg = EnvConfig(
        file_name=cfg.file_name,
        behavior_name=cfg.behavior_name,
        no_graphics=cfg.no_graphics,
        obs_size=cfg.obs_dim,
        lidar_dim=cfg.lidar_dim,
        reach_goal_radius=cfg.reach_goal_radius,
        max_steps=cfg.max_steps,
        progress_gain=cfg.progress_gain,
        time_penalty=cfg.time_penalty,
        collision_penalty=cfg.collision_penalty,
        success_bonus=cfg.success_bonus,
        timeout_penalty=cfg.timeout_penalty,
        near_obstacle_threshold=cfg.near_obstacle_threshold,
        near_obstacle_penalty=cfg.near_obstacle_penalty,
        action_l2_penalty=cfg.action_l2_penalty,
    )

    print(f"Loading checkpoint from: {cfg.checkpoint_path}")
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)

    model = ActorCritic(cfg.obs_dim, cfg.action_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = UnityNavEnv(env_cfg)

    results = []
    for ep in range(cfg.num_eval_episodes):
        result = evaluate_one_episode(env, model, device)
        results.append(result)
        print(f"[PPO-EVAL][{ep+1:03d}] {result}")

    env.close()
    summarize_results(results)


if __name__ == "__main__":
    main()
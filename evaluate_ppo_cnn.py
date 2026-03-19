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

    # 改成你想评估的 CNN-PPO checkpoint
    checkpoint_path: str = r"./checkpoints/cnn_ppo/second_test/ppo_update_2700.pt"

    # 评估 episode 数
    num_eval_episodes: int = 100

    # 是否显示 Unity 窗口
    no_graphics: bool = False

    # 观测/动作维度
    obs_dim: int = 187
    lidar_dim: int = 180
    low_dim: int = 7
    action_dim: int = 2

    # 环境参数：建议和训练保持一致
    reach_goal_radius: float = 0.5
    max_steps: int = 250

    # reward 参数：与 train_ppo_cnn.py 一致
    progress_gain: float = 1.5
    time_penalty: float = -0.002
    collision_penalty: float = -2.0
    success_bonus: float = 15.0
    timeout_penalty: float = -2.0
    near_obstacle_threshold: float = 0.4
    near_obstacle_penalty: float = -0.005
    action_l2_penalty: float = -0.0005


# =========================
# CNN Actor-Critic
# =========================
class CNNActorCritic(nn.Module):
    """
    观测拆分:
      lidar   = obs[:, :180]
      low_dim = obs[:, 180:]   # 共 7 维
    """
    def __init__(self, lidar_dim: int, low_dim: int, action_dim: int):
        super().__init__()

        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, lidar_dim)
            lidar_feat_dim = self.lidar_encoder(dummy).shape[1]

        self.lidar_fc = nn.Sequential(
            nn.Linear(lidar_feat_dim, 128),
            nn.Tanh(),
        )

        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(256, 1)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        lidar = obs[:, :180]
        low_dim = obs[:, 180:]

        lidar = lidar.unsqueeze(1)  # (B, 1, 180)
        lidar_feat = self.lidar_encoder(lidar)
        lidar_feat = self.lidar_fc(lidar_feat)

        low_feat = self.low_encoder(low_dim)

        feat = torch.cat([lidar_feat, low_feat], dim=-1)
        feat = self.fusion(feat)
        return feat

    def forward(self, obs: torch.Tensor):
        feat = self.encode_obs(obs)
        mean = self.actor_mean(feat)
        value = self.critic(feat)
        logstd = self.actor_logstd.expand_as(mean)
        return mean, logstd, value

    def get_deterministic_action(self, obs: torch.Tensor):
        mean, _, value = self(obs)
        return mean, value.squeeze(-1)


# =========================
# Evaluation Helpers
# =========================
def evaluate_one_episode(env: UnityNavEnv, model: CNNActorCritic, device: torch.device):
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

    print("\n=== CNN-PPO Evaluation Summary ===")
    print(f"success_rate        : {success_rate:.3f}")
    print(f"collision_rate      : {collision_rate:.3f}")
    print(f"timeout_rate        : {timeout_rate:.3f}")
    print(f"avg_len             : {avg_len:.2f}")
    print(f"avg_return          : {avg_ret:.3f}")
    print(f"avg_final_goal_dist : {avg_final_goal_dist:.3f}")


# =========================
# Main
# =========================
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

    model = CNNActorCritic(
        lidar_dim=cfg.lidar_dim,
        low_dim=cfg.low_dim,
        action_dim=cfg.action_dim,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    env = UnityNavEnv(env_cfg)

    results = []
    for ep in range(cfg.num_eval_episodes):
        result = evaluate_one_episode(env, model, device)
        results.append(result)
        print(f"[CNN-PPO-EVAL][{ep+1:03d}] {result}")

    env.close()
    summarize_results(results)


if __name__ == "__main__":
    main()
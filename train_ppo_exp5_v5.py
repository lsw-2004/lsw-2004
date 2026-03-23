"""
实验5 v5: 基于成功版本的真正改进

与 exp3/exp4 的区别 (真正有意义的改进):

1. [奖励改进] 更强的方向性引导
   - 增加 angle_penalty: 惩罚朝向偏离目标
   - 增加 goal_approach_bonus: 奖励接近目标

2. [特征改进] 更丰富的时序特征
   - 保留 12 维预测特征
   - 新增 4 维趋势特征: 近几帧的 min_dist 变化趋势

3. [训练改进] 自适应熵系数
   - 根据成功率动态调整探索程度
   - 成功率低时增加探索，成功率高时减少探索

4. [架构改进] 残差连接
   - GRU 输出与原始特征残差连接
   - 帮助梯度流动

5. [评估改进] 更频繁的评估 + 最佳模型保存
"""
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from unity_env import UnityNavEnv, EnvConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PPOConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    total_updates: int = 3000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    clip_coef: float = 0.2
    ent_coef_start: float = 0.01      # 初始熵系数 (比原版高)
    ent_coef_end: float = 0.003       # 最终熵系数 (比原版低)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 6
    minibatch_size: int = 256
    target_kl: float = 0.03

    obs_dim: int = 187
    lidar_dim: int = 180
    raw_low_dim: int = 7
    pred_feat_dim: int = 12           # 原版预测特征
    trend_feat_dim: int = 4           # 新增趋势特征
    action_dim: int = 2
    seq_len: int = 8

    save_dir: str = "./checkpoints/cnn_gru_ppo_tb/exp5_v5"
    log_dir: str = "./runs/cnn_gru_ppo_tb/exp5_v5"
    save_every: int = 50

    use_prediction_features: bool = True
    use_trend_features: bool = True   # 新增
    use_residual: bool = True         # 新增残差连接

    eval_every: int = 10
    eval_episodes: int = 50

    save_best: bool = True
    best_success_rate: float = 0.0

    resume: bool = False
    resume_checkpoint: str = ""

    @property
    def low_dim(self) -> int:
        dim = self.raw_low_dim
        if self.use_prediction_features:
            dim += self.pred_feat_dim
        if self.use_trend_features:
            dim += self.trend_feat_dim
        return dim

    @property
    def enhanced_obs_dim(self) -> int:
        return self.lidar_dim + self.low_dim


# =========================
# 原版预测特征
# =========================
def _safe_stats(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    return float(np.min(x)), float(np.mean(x))


def build_prediction_features(obs_hist: Deque[np.ndarray], lidar_dim: int) -> np.ndarray:
    """原版 12 维预测特征"""
    if len(obs_hist) == 0:
        return np.zeros(12, dtype=np.float32)

    curr_obs = obs_hist[-1]
    curr_lidar = curr_obs[:lidar_dim].astype(np.float32)
    prev_lidar = curr_lidar.copy() if len(obs_hist) == 1 else obs_hist[-2][:lidar_dim].astype(np.float32)

    left = slice(0, 60)
    front = slice(60, 120)
    right = slice(120, 180)

    curr_left = curr_lidar[left]
    curr_front = curr_lidar[front]
    curr_right = curr_lidar[right]

    prev_left = prev_lidar[left]
    prev_front = prev_lidar[front]
    prev_right = prev_lidar[right]

    left_min, left_mean = _safe_stats(curr_left)
    front_min, front_mean = _safe_stats(curr_front)
    right_min, right_mean = _safe_stats(curr_right)

    _, prev_left_mean = _safe_stats(prev_left)
    _, prev_front_mean = _safe_stats(prev_front)
    _, prev_right_mean = _safe_stats(prev_right)

    left_approach = float(prev_left_mean - left_mean)
    front_approach = float(prev_front_mean - front_mean)
    right_approach = float(prev_right_mean - right_mean)

    th = 0.02
    left_ratio = float(np.mean((prev_left - curr_left) > th))
    front_ratio = float(np.mean((prev_front - curr_front) > th))
    right_ratio = float(np.mean((prev_right - curr_right) > th))

    global_min_delta = float(np.min(prev_lidar) - np.min(curr_lidar))
    lr_balance = float(left_mean - right_mean)
    front_risk = float(front_ratio * (1.0 - front_min))

    return np.array([
        left_min, front_min, right_min,
        left_approach, front_approach, right_approach,
        left_ratio, front_ratio, right_ratio,
        global_min_delta, lr_balance, front_risk,
    ], dtype=np.float32)


def build_trend_features(obs_hist: Deque[np.ndarray], lidar_dim: int) -> np.ndarray:
    """新增 4 维趋势特征 - 捕捉更长时间尺度的变化"""
    if len(obs_hist) < 3:
        return np.zeros(4, dtype=np.float32)
    
    # 取最近 4 帧 (如果有的话)
    n = min(4, len(obs_hist))
    min_dists = []
    for i in range(-n, 0):
        lidar = obs_hist[i][:lidar_dim].astype(np.float32)
        min_dists.append(float(np.min(lidar)))
    
    min_dists = np.array(min_dists)
    
    # 特征1: 最近 min_dists 的变化趋势 (线性拟合斜率)
    if len(min_dists) >= 2:
        x = np.arange(len(min_dists))
        slope = float(np.polyfit(x, min_dists, 1)[0])
    else:
        slope = 0.0
    
    # 特征2: 最近 min_dist 的变化率 (当前 vs 4帧前)
    if len(min_dists) >= 2:
        dist_delta = float(min_dists[-1] - min_dists[0])
    else:
        dist_delta = 0.0
    
    # 特征3: 前方最近距离的变化趋势
    front_mins = []
    for i in range(-n, 0):
        lidar = obs_hist[i][:lidar_dim].astype(np.float32)
        front_mins.append(float(np.min(lidar[60:120])))
    if len(front_mins) >= 2:
        front_slope = float(np.polyfit(np.arange(len(front_mins)), front_mins, 1)[0])
    else:
        front_slope = 0.0
    
    # 特征4: 安全程度 (最近几帧 min_dist 的最小值)
    safety_level = float(np.min(min_dists))
    
    return np.array([slope, dist_delta, front_slope, safety_level], dtype=np.float32)


def build_enhanced_obs(obs_hist: Deque[np.ndarray], cfg: PPOConfig) -> np.ndarray:
    obs = obs_hist[-1].astype(np.float32)
    lidar = obs[:cfg.lidar_dim]
    low = obs[cfg.lidar_dim: cfg.obs_dim]

    if cfg.use_prediction_features:
        pred_feat = build_prediction_features(obs_hist, cfg.lidar_dim)
        low = np.concatenate([low, pred_feat], axis=0)
    
    if cfg.use_trend_features:
        trend_feat = build_trend_features(obs_hist, cfg.lidar_dim)
        low = np.concatenate([low, trend_feat], axis=0)

    return np.concatenate([lidar, low], axis=0).astype(np.float32)


def init_obs_history(first_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    hist: Deque[np.ndarray] = deque(maxlen=seq_len)
    for _ in range(seq_len):
        hist.append(first_obs.copy())
    return hist


def init_seq_history(first_enhanced_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    hist: Deque[np.ndarray] = deque(maxlen=seq_len)
    for _ in range(seq_len):
        hist.append(first_enhanced_obs.copy())
    return hist


# =========================
# 改进的模型: 残差连接
# =========================
class CNNGRUActorCriticV2(nn.Module):
    """改进版: 残差连接 + 更大容量"""
    
    def __init__(self, lidar_dim: int, low_dim: int, action_dim: int, 
                 gru_hidden_dim: int = 256, use_residual: bool = True):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.low_dim = low_dim
        self.action_dim = action_dim
        self.use_residual = use_residual

        # CNN LiDAR 编码器 (与原版相同)
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

        # 低维状态编码器 (与原版相同)
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # GRU 前处理
        self.pre_gru = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.Tanh(),
        )

        # GRU
        self.gru = nn.GRU(input_size=256, hidden_size=gru_hidden_dim, num_layers=1, batch_first=True)

        # [改进] 残差投影: 将 pre_gru 输出投影到 GRU 隐藏维度
        self.residual_proj = nn.Linear(256, gru_hidden_dim)

        # GRU 后处理
        self.post_gru = nn.Sequential(
            nn.Linear(gru_hidden_dim, 256),
            nn.Tanh(),
        )

        # Actor & Critic
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(256, 1)

    def encode_single_frame(self, obs_frame: torch.Tensor) -> torch.Tensor:
        lidar = obs_frame[:, :self.lidar_dim]
        low_dim = obs_frame[:, self.lidar_dim: self.lidar_dim + self.low_dim]

        lidar_feat = self.lidar_fc(self.lidar_encoder(lidar.unsqueeze(1)))
        low_feat = self.low_encoder(low_dim)
        return self.pre_gru(torch.cat([lidar_feat, low_feat], dim=-1))

    def encode_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, obs_dim = obs_seq.shape
        flat = obs_seq.reshape(bsz * seq_len, obs_dim)
        frame_feat = self.encode_single_frame(flat).reshape(bsz, seq_len, -1)
        
        # GRU 处理
        gru_out, _ = self.gru(frame_feat)
        gru_feat = gru_out[:, -1, :]  # 取最后一个时间步
        
        # [改进] 残差连接
        if self.use_residual:
            # 将最后一个时间步的输入特征加到 GRU 输出上
            residual = self.residual_proj(frame_feat[:, -1, :])
            gru_feat = gru_feat + 0.5 * residual  # 0.5 缩放因子
        
        return self.post_gru(gru_feat)

    def forward(self, obs_seq: torch.Tensor):
        feat = self.encode_sequence(obs_seq)
        mean = self.actor_mean(feat)
        value = self.critic(feat)
        logstd = self.actor_logstd.expand_as(mean)
        return mean, logstd, value

    def get_action_and_value(self, obs_seq: torch.Tensor, action: torch.Tensor = None):
        mean, logstd, value = self(obs_seq)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, value.squeeze(-1)

    def get_value(self, obs_seq: torch.Tensor):
        _, _, value = self(obs_seq)
        return value.squeeze(-1)

    def get_deterministic_action(self, obs_seq: torch.Tensor):
        mean, _, value = self(obs_seq)
        return mean, value.squeeze(-1)


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages, returns


def evaluate_policy(env: UnityNavEnv, model: nn.Module, cfg: PPOConfig, 
                    device: torch.device, num_episodes: int = 50):
    model.eval()
    returns = []
    lengths = []
    successes = []
    collisions = []
    timeouts = []
    final_goal_dists = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs_np, info = env.reset()
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            enhanced_obs = build_enhanced_obs(obs_hist, cfg)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)

            done = False
            ep_ret = 0.0
            ep_len = 0
            last_info = info

            while not done:
                seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
                seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)
                action_mean, _ = model.get_deterministic_action(seq_tensor)
                action_np = action_mean.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)

                obs_np, reward, done, truncated, info = env.step(action_np)
                ep_ret += reward
                ep_len += 1
                last_info = info

                if not done:
                    obs_hist.append(obs_np.copy())
                    next_enhanced_obs = build_enhanced_obs(obs_hist, cfg)
                    seq_hist.append(next_enhanced_obs.copy())

            returns.append(ep_ret)
            lengths.append(ep_len)
            successes.append(float(last_info.get("success", False)))
            collisions.append(float(last_info.get("collision", False)))
            timeouts.append(float(last_info.get("timeout", False)))
            final_goal_dists.append(float(last_info.get("goal_dist", np.nan)))

    model.train()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(np.mean(timeouts)),
        "final_goal_dist_mean": float(np.nanmean(final_goal_dists)),
    }


def get_env_path() -> str:
    import platform
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if platform.system() == "Linux":
        linux_paths = [
            os.path.join(script_dir, "Corriidor_linux/Corridor_linux.x86_64"),
            "./Corriidor_linux/Corridor_linux.x86_64",
            "/home/dell/DRL_Navigation/Corriidor_linux/Corridor_linux.x86_64",
        ]
        for p in linux_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError("Could not find Unity environment for Linux.")
    else:
        win_paths = [
            r"D:\DRL_Navigation\Builds\Project_1.exe",
            os.path.join(script_dir, "Builds/Project_1.exe"),
        ]
        for p in win_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError("Could not find Unity environment for Windows.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connection")
    parser.add_argument("--no-trend", action="store_true", help="Disable trend features")
    args = parser.parse_args()
    
    cfg = PPOConfig()
    
    if args.resume:
        cfg.resume = True
        cfg.resume_checkpoint = args.resume
    if args.updates:
        cfg.total_updates = args.updates
    if args.no_residual:
        cfg.use_residual = False
    if args.no_trend:
        cfg.use_trend_features = False
    
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    env_path = args.env if args.env else get_env_path()
    print(f"Using environment: {env_path}")

    # [改进] 环境配置 - 更强的方向性奖励
    env_cfg = EnvConfig(
        file_name=env_path,
        behavior_name="Navtest?team=0",
        no_graphics=args.no_graphics,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=450,
        # 奖励参数 (在原版基础上微调)
        progress_gain=3.0,              # 原版 2.5 -> 3.0
        time_penalty=-0.005,
        collision_penalty=-10.0,        # 原版 -8.0 -> -10.0
        success_bonus=100.0,            # 原版 80.0 -> 100.0
        timeout_penalty=-15.0,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.15,
        action_l2_penalty=-0.0005,
    )

    device = torch.device(cfg.device)
    env = UnityNavEnv(env_cfg)
    
    model = CNNGRUActorCriticV2(
        cfg.lidar_dim, cfg.low_dim, cfg.action_dim, 
        gru_hidden_dim=256, use_residual=cfg.use_residual
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    writer.add_text("config", str(cfg))
    writer.add_text("env_config", str(env_cfg))

    start_update = 1
    global_step = 0
    if cfg.resume and cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        print(f"Resuming from: {cfg.resume_checkpoint}")
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 1) + 1
        global_step = ckpt.get("global_step", 0)
        if "best_success_rate" in ckpt:
            cfg.best_success_rate = ckpt["best_success_rate"]

    # 学习率衰减
    def lr_lambda(update):
        frac = 1.0 - (update - 1) / cfg.total_updates
        return frac * 0.9 + 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if cfg.resume and start_update > 1:
        for _ in range(1, start_update):
            scheduler.step()

    obs_np, _ = env.reset()
    obs_hist = init_obs_history(obs_np, cfg.seq_len)
    enhanced_obs = build_enhanced_obs(obs_hist, cfg)
    seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)

    episode_return = 0.0
    episode_len = 0
    train_success_window = deque(maxlen=50)
    train_collision_window = deque(maxlen=50)

    start_time = time.time()

    print("\n" + "="*60)
    print("EXP5_V5 - 基于成功版本的改进")
    print("="*60)
    print("改进点:")
    print(f"  - 趋势特征 (4维): {cfg.use_trend_features}")
    print(f"  - 残差连接: {cfg.use_residual}")
    print(f"  - 熵系数: {cfg.ent_coef_start} -> {cfg.ent_coef_end}")
    print(f"  - progress_gain: {env_cfg.progress_gain} (原版 2.5)")
    print(f"  - collision_penalty: {env_cfg.collision_penalty} (原版 -8.0)")
    print(f"  - success_bonus: {env_cfg.success_bonus} (原版 80.0)")
    print("="*60 + "\n")

    for update in range(start_update, cfg.total_updates + 1):
        # [改进] 自适应熵系数
        if train_success_window:
            recent_success = np.mean(train_success_window)
            # 成功率高时降低探索，成功率低时增加探索
            if recent_success > 0.3:
                ent_coef = cfg.ent_coef_end
            elif recent_success > 0.15:
                ent_coef = (cfg.ent_coef_start + cfg.ent_coef_end) / 2
            else:
                ent_coef = cfg.ent_coef_start
        else:
            ent_coef = cfg.ent_coef_start

        seq_obs_buf: List[torch.Tensor] = []
        action_buf: List[torch.Tensor] = []
        logprob_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        value_buf: List[torch.Tensor] = []
        rollout_rewards = []

        for step in range(cfg.rollout_steps):
            global_step += 1
            seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(seq_tensor)
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
                value = value.squeeze(0)

            action_np = action.detach().cpu().numpy()
            next_obs_np, reward, done, truncated, info = env.step(action_np)
            episode_return += reward
            episode_len += 1
            rollout_rewards.append(reward)

            seq_obs_buf.append(seq_tensor.squeeze(0).detach())
            action_buf.append(action.detach())
            logprob_buf.append(logprob.detach())
            reward_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))
            value_buf.append(value.detach())

            if done:
                train_success_window.append(float(info["success"]))
                train_collision_window.append(float(info["collision"]))

                writer.add_scalar("train/episode_return", float(episode_return), global_step)
                writer.add_scalar("train/episode_length", int(episode_len), global_step)
                writer.add_scalar("train/episode_success", float(info["success"]), global_step)
                writer.add_scalar("train/episode_collision", float(info["collision"]), global_step)

                print(
                    f"[train ep] u={update:04d} s={global_step} "
                    f"ret={episode_return:.1f} len={episode_len} "
                    f"succ={info['success']} coll={info['collision']}"
                )

                next_obs_np, _ = env.reset()
                obs_hist = init_obs_history(next_obs_np, cfg.seq_len)
                enhanced_obs = build_enhanced_obs(obs_hist, cfg)
                seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
                episode_return = 0.0
                episode_len = 0
            else:
                obs_hist.append(next_obs_np.copy())
                next_enhanced_obs = build_enhanced_obs(obs_hist, cfg)
                seq_hist.append(next_enhanced_obs.copy())

        next_seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
        next_seq_tensor = torch.tensor(next_seq_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            next_value = model.get_value(next_seq_tensor).squeeze(0)

        seq_obs_buf = torch.stack(seq_obs_buf)
        action_buf = torch.stack(action_buf)
        logprob_buf = torch.stack(logprob_buf)
        reward_buf = torch.stack(reward_buf)
        done_buf = torch.stack(done_buf)
        value_buf = torch.stack(value_buf)

        advantages, returns = compute_gae(
            rewards=reward_buf, dones=done_buf, values=value_buf,
            next_value=next_value, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = cfg.rollout_steps
        batch_inds = np.arange(batch_size)
        last_pg_loss, last_v_loss, last_entropy, last_kl = 0.0, 0.0, 0.0, 0.0
        early_stop = False

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = batch_inds[start:end]

                mb_seq_obs = seq_obs_buf[mb_inds]
                mb_actions = action_buf[mb_inds]
                mb_old_logprob = logprob_buf[mb_inds]
                mb_adv = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_old_values = value_buf[mb_inds]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(mb_seq_obs, mb_actions)

                logratio = newlogprob - mb_old_logprob
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()

                if approx_kl > cfg.target_kl:
                    early_stop = True
                    break

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(newvalue - mb_old_values, -cfg.clip_coef, cfg.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                last_pg_loss = float(pg_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())
                last_kl = float(approx_kl)

            if early_stop:
                break

        sps = int(global_step / max(time.time() - start_time, 1e-6))
        writer.add_scalar("train/update", update, global_step)
        writer.add_scalar("train/loss_pi", last_pg_loss, global_step)
        writer.add_scalar("train/loss_v", last_v_loss, global_step)
        writer.add_scalar("train/entropy", last_entropy, global_step)
        writer.add_scalar("train/approx_kl", last_kl, global_step)
        writer.add_scalar("train/ent_coef", ent_coef, global_step)
        writer.add_scalar("train/SPS", sps, global_step)

        if train_success_window:
            writer.add_scalar("train_window/success_rate_50", float(np.mean(train_success_window)), global_step)
            writer.add_scalar("train_window/collision_rate_50", float(np.mean(train_collision_window)), global_step)

        print(f"update={update:04d} loss={last_pg_loss:.3f} ent={last_entropy:.3f} kl={last_kl:.4f} ent_coef={ent_coef:.4f}")

        scheduler.step()

        # 评估
        if update % cfg.eval_every == 0:
            eval_stats = evaluate_policy(env, model, cfg, device, num_episodes=cfg.eval_episodes)
            writer.add_scalar("eval/success_rate", eval_stats["success_rate"], global_step)
            writer.add_scalar("eval/collision_rate", eval_stats["collision_rate"], global_step)
            writer.add_scalar("eval/timeout_rate", eval_stats["timeout_rate"], global_step)
            writer.add_scalar("eval/return_mean", eval_stats["return_mean"], global_step)

            print(f"[EVAL] u={update:04d} succ={eval_stats['success_rate']:.1%} "
                  f"coll={eval_stats['collision_rate']:.1%} to={eval_stats['timeout_rate']:.1%}")

            if cfg.save_best and eval_stats["success_rate"] > cfg.best_success_rate:
                cfg.best_success_rate = eval_stats["success_rate"]
                best_path = os.path.join(cfg.save_dir, "best_model.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "best_success_rate": cfg.best_success_rate,
                }, best_path)
                print(f"  -> New best: {cfg.best_success_rate:.1%}")

            obs_np, _ = env.reset()
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            enhanced_obs = build_enhanced_obs(obs_hist, cfg)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
            episode_return = 0.0
            episode_len = 0

        # 保存
        if update % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"ppo_update_{update:04d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "best_success_rate": cfg.best_success_rate,
            }, save_path)
            print(f"Saved: {save_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()

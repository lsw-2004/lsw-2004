"""
实验5 v4: 简化版 - 专注于核心问题

关键改进:
1. 简化模型: 使用 GRU 替代 Transformer (更稳定)
2. 移除动态检测: 当前实现有问题，先移除
3. 密集奖励: 增加目标导向奖励
4. 更强的熵正则: 防止过早收敛
5. 完整日志: 包含 timeout 比例
"""
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    # 训练参数
    total_updates: int = 3000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # 优化器参数
    lr: float = 3e-4
    clip_coef: float = 0.2
    
    # Entropy - 关键: 保持足够的探索
    ent_coef: float = 0.01           # 固定值，不衰减
    entropy_target: float = 1.0      # 目标 entropy
    entropy_coef_max: float = 0.05   # 最大 entropy 系数
    
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 4           # 减少 epoch 防止过拟合
    minibatch_size: int = 256
    target_kl: float = 0.02

    # 观测参数 - 简化
    obs_dim: int = 187
    lidar_dim: int = 180
    low_dim: int = 7                 # 只用原始低维状态
    
    action_dim: int = 2
    seq_len: int = 8                 # 缩短序列

    # 保存路径
    save_dir: str = "./checkpoints/cnn_gru_ppo_tb/exp5_v4"
    log_dir: str = "./runs/cnn_gru_ppo_tb/exp5_v4"
    save_every: int = 100

    # 评估
    eval_every: int = 50
    eval_episodes: int = 50

    # 最佳模型
    save_best: bool = True
    best_success_rate: float = 0.0

    resume: bool = False
    resume_checkpoint: str = ""


# =========================
# 简化模型: CNN + GRU
# =========================
class SimpleActorCritic(nn.Module):
    """简化的 Actor-Critic 模型"""
    
    def __init__(self, lidar_dim: int, low_dim: int, action_dim: int, seq_len: int = 8):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.low_dim = low_dim
        self.action_dim = action_dim
        self.seq_len = seq_len

        # LiDAR 编码器 - 简化
        self.lidar_encoder = nn.Sequential(
            nn.Linear(lidar_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 低维状态编码器
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, 64),
            nn.ReLU(),
        )

        # GRU 时序编码
        self.gru = nn.GRU(
            input_size=128 + 64,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs_seq: torch.Tensor):
        """
        obs_seq: [batch, seq_len, obs_dim]
        obs_dim = lidar_dim + low_dim
        """
        bsz, seq_len, _ = obs_seq.shape
        
        # 编码每一帧
        lidar = obs_seq[:, :, :self.lidar_dim]
        low = obs_seq[:, :, self.lidar_dim:self.lidar_dim + self.low_dim]
        
        # 展平处理
        lidar_flat = lidar.reshape(bsz * seq_len, self.lidar_dim)
        low_flat = low.reshape(bsz * seq_len, self.low_dim)
        
        lidar_feat = self.lidar_encoder(lidar_flat)
        low_feat = self.low_encoder(low_flat)
        
        frame_feat = torch.cat([lidar_feat, low_feat], dim=-1)
        frame_feat = frame_feat.reshape(bsz, seq_len, -1)
        
        # GRU
        _, h = self.gru(frame_feat)
        temporal_feat = h[-1]  # 最后一层隐藏状态
        
        # 输出
        mean = self.actor(temporal_feat)
        value = self.critic(temporal_feat)
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


# =========================
# GAE
# =========================
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


def init_obs_history(first_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    hist: Deque[np.ndarray] = deque(maxlen=seq_len)
    for _ in range(seq_len):
        hist.append(first_obs.copy())
    return hist


# =========================
# 评估函数
# =========================
def evaluate_policy(
    env: UnityNavEnv, 
    model: SimpleActorCritic, 
    cfg: PPOConfig, 
    device: torch.device, 
    num_episodes: int = 50
):
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

            done = False
            ep_ret = 0.0
            ep_len = 0
            last_info = info

            while not done:
                seq_np = np.stack(list(obs_hist), axis=0).astype(np.float32)
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


# =========================
# 主训练函数
# =========================
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
    
    parser = argparse.ArgumentParser(description="Train PPO exp5_v4 - simplified version")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--updates", type=int, default=None)
    args = parser.parse_args()
    
    cfg = PPOConfig()
    
    if args.resume:
        cfg.resume = True
        cfg.resume_checkpoint = args.resume
    if args.updates:
        cfg.total_updates = args.updates
    
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    env_path = args.env if args.env else get_env_path()
    print(f"Using environment: {env_path}")

    # 环境配置 - 关键调整
    env_cfg = EnvConfig(
        file_name=env_path,
        behavior_name="Navtest?team=0",
        no_graphics=args.no_graphics,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=450,
        # 奖励调整: 更强的引导
        progress_gain=5.0,              # 增加进度奖励
        time_penalty=-0.01,             # 增加时间惩罚
        collision_penalty=-30.0,        # 更强的碰撞惩罚
        success_bonus=100.0,
        timeout_penalty=-10.0,          # 减少 timeout 惩罚 (相对于碰撞)
        near_obstacle_threshold=1.0,    # 更大的预警范围
        near_obstacle_penalty=-0.5,     # 更强的近障碍惩罚
        action_l2_penalty=-0.001,
    )

    device = torch.device(cfg.device)
    env = UnityNavEnv(env_cfg)
    
    model = SimpleActorCritic(
        cfg.lidar_dim, 
        cfg.low_dim, 
        cfg.action_dim,
        cfg.seq_len
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    
    # 记录配置
    writer.add_text("config", str(cfg))
    writer.add_text("env_config", str(env_cfg))

    start_update = 1
    global_step = 0
    if cfg.resume and cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        print(f"Resuming from checkpoint: {cfg.resume_checkpoint}")
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 1) + 1
        global_step = ckpt.get("global_step", 0)
        if "best_success_rate" in ckpt:
            cfg.best_success_rate = ckpt["best_success_rate"]

    obs_np, _ = env.reset()
    obs_hist = init_obs_history(obs_np, cfg.seq_len)

    episode_return = 0.0
    episode_len = 0
    train_returns_window = deque(maxlen=100)
    train_success_window = deque(maxlen=100)
    train_collision_window = deque(maxlen=100)
    train_timeout_window = deque(maxlen=100)
    
    # 追踪 entropy 用于动态调整
    entropy_history = deque(maxlen=20)
    
    start_time = time.time()

    print("\n" + "="*60)
    print("EXP5_V4 简化版训练")
    print("="*60)
    print(f"关键参数:")
    print(f"  - 固定 entropy 系数: {cfg.ent_coef}")
    print(f"  - 碰撞惩罚: {env_cfg.collision_penalty}")
    print(f"  - 进度奖励: {env_cfg.progress_gain}")
    print(f"  - 学习率: {cfg.lr}")
    print("="*60 + "\n")

    for update in range(start_update, cfg.total_updates + 1):
        obs_buf: List[torch.Tensor] = []
        action_buf: List[torch.Tensor] = []
        logprob_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        value_buf: List[torch.Tensor] = []

        for step in range(cfg.rollout_steps):
            global_step += 1
            seq_np = np.stack(list(obs_hist), axis=0).astype(np.float32)
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, logprob, entropy, value = model.get_action_and_value(seq_tensor)
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
                value = value.squeeze(0)

            action_np = action.detach().cpu().numpy()
            next_obs_np, reward, done, truncated, info = env.step(action_np)
            
            episode_return += reward
            episode_len += 1

            obs_buf.append(seq_tensor.squeeze(0).detach())
            action_buf.append(action.detach())
            logprob_buf.append(logprob.detach())
            reward_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))
            value_buf.append(value.detach())
            
            entropy_history.append(float(entropy.mean().item()))

            if done:
                train_returns_window.append(float(episode_return))
                train_success_window.append(float(info["success"]))
                train_collision_window.append(float(info["collision"]))
                train_timeout_window.append(float(info["timeout"]))

                writer.add_scalar("train/episode_return", float(episode_return), global_step)
                writer.add_scalar("train/episode_length", int(episode_len), global_step)
                writer.add_scalar("train/episode_success", float(info["success"]), global_step)
                writer.add_scalar("train/episode_collision", float(info["collision"]), global_step)
                writer.add_scalar("train/episode_timeout", float(info["timeout"]), global_step)

                print(
                    f"[ep] u={update:04d} s={global_step} "
                    f"ret={episode_return:.1f} len={episode_len} "
                    f"succ={info['success']} coll={info['collision']} to={info['timeout']}"
                )

                next_obs_np, _ = env.reset()
                obs_hist = init_obs_history(next_obs_np, cfg.seq_len)
                episode_return = 0.0
                episode_len = 0
            else:
                obs_hist.append(next_obs_np.copy())

        # GAE
        seq_np = np.stack(list(obs_hist), axis=0).astype(np.float32)
        seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            next_value = model.get_value(seq_tensor).squeeze(0)

        obs_buf = torch.stack(obs_buf)
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

        # 动态调整 entropy 系数
        if len(entropy_history) > 0:
            avg_entropy = np.mean(list(entropy_history))
            if avg_entropy < cfg.entropy_target:
                current_ent_coef = min(cfg.ent_coef * 1.5, cfg.entropy_coef_max)
            else:
                current_ent_coef = cfg.ent_coef
        else:
            current_ent_coef = cfg.ent_coef

        # PPO 更新
        batch_size = cfg.rollout_steps
        batch_inds = np.arange(batch_size)
        last_pg_loss, last_v_loss, last_entropy = 0.0, 0.0, 0.0
        early_stop = False

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = batch_inds[start:end]

                mb_obs = obs_buf[mb_inds]
                mb_actions = action_buf[mb_inds]
                mb_old_logprob = logprob_buf[mb_inds]
                mb_adv = advantages[mb_inds]
                mb_returns = returns[mb_inds]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(mb_obs, mb_actions)

                logratio = newlogprob - mb_old_logprob
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()

                if approx_kl > cfg.target_kl:
                    early_stop = True
                    break

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * (newvalue - mb_returns) ** 2
                v_loss = v_loss.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - current_ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                last_pg_loss = float(pg_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())

            if early_stop:
                break

        # 日志
        sps = int(global_step / max(time.time() - start_time, 1e-6))
        writer.add_scalar("train/update", update, global_step)
        writer.add_scalar("train/loss_pi", last_pg_loss, global_step)
        writer.add_scalar("train/loss_v", last_v_loss, global_step)
        writer.add_scalar("train/entropy", last_entropy, global_step)
        writer.add_scalar("train/ent_coef", current_ent_coef, global_step)
        writer.add_scalar("train/SPS", sps, global_step)

        if train_returns_window:
            writer.add_scalar("train_window/success_rate_100", float(np.mean(train_success_window)), global_step)
            writer.add_scalar("train_window/collision_rate_100", float(np.mean(train_collision_window)), global_step)
            writer.add_scalar("train_window/timeout_rate_100", float(np.mean(train_timeout_window)), global_step)
            writer.add_scalar("train_window/return_mean_100", float(np.mean(train_returns_window)), global_step)

        print(f"update={update:04d} loss={last_pg_loss:.3f} ent={last_entropy:.3f} "
              f"ent_coef={current_ent_coef:.4f} succ={np.mean(train_success_window):.2%}")

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

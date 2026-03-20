import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from unity_env import UnityNavEnv, EnvConfig


# =========================
# Utils
# =========================
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

    total_updates: int = 1000
    rollout_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 10
    minibatch_size: int = 128

    obs_dim: int = 187
    lidar_dim: int = 180
    raw_low_dim: int = 7
    pred_feat_dim: int = 12
    action_dim: int = 2
    seq_len: int = 8

    save_dir: str = "./checkpoints/cnn_gru_ppo_tb/exp1"
    log_dir: str = "./runs/cnn_gru_ppo_tb/exp1"
    save_every: int = 50

    use_prediction_features: bool = True

    # 在线评估
    eval_every: int = 10          # 每多少个 update 做一次评估
    eval_episodes: int = 50        # 每次评估跑多少回合
    eval_deterministic: bool = True

    @property
    def low_dim(self) -> int:
        return self.raw_low_dim + (self.pred_feat_dim if self.use_prediction_features else 0)

    @property
    def enhanced_obs_dim(self) -> int:
        return self.lidar_dim + self.low_dim


# =========================
# LiDAR motion features
# =========================
def _safe_stats(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    return float(np.min(x)), float(np.mean(x))


def build_prediction_features(obs_hist: Deque[np.ndarray], lidar_dim: int) -> np.ndarray:
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

    feat = np.array(
        [
            left_min, front_min, right_min,
            left_approach, front_approach, right_approach,
            left_ratio, front_ratio, right_ratio,
            global_min_delta, lr_balance, front_risk,
        ],
        dtype=np.float32,
    )
    return feat


def build_enhanced_obs(obs_hist: Deque[np.ndarray], cfg: PPOConfig) -> np.ndarray:
    obs = obs_hist[-1].astype(np.float32)
    lidar = obs[:cfg.lidar_dim]
    low = obs[cfg.lidar_dim: cfg.obs_dim]

    if cfg.use_prediction_features:
        pred_feat = build_prediction_features(obs_hist, cfg.lidar_dim)
        low = np.concatenate([low, pred_feat], axis=0)

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


class CNNGRUActorCritic(nn.Module):
    def __init__(self, lidar_dim: int, low_dim: int, action_dim: int, gru_hidden_dim: int = 256):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.low_dim = low_dim
        self.action_dim = action_dim

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

        self.pre_gru = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.Tanh(),
        )

        self.gru = nn.GRU(input_size=256, hidden_size=gru_hidden_dim, num_layers=1, batch_first=True)

        self.post_gru = nn.Sequential(
            nn.Linear(gru_hidden_dim, 256),
            nn.Tanh(),
        )

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
        gru_out, _ = self.gru(frame_feat)
        return self.post_gru(gru_out[:, -1, :])

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


def evaluate_policy(env: UnityNavEnv, model: CNNGRUActorCritic, cfg: PPOConfig, device: torch.device, num_episodes: int = 5):
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


def main():
    cfg = PPOConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    env_cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\Builds\Project_1.exe",   # 改成你的真实路径
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=350,
        progress_gain=1.5,
        time_penalty=-0.002,
        collision_penalty=-2.0,
        success_bonus=15.0,
        timeout_penalty=-2.2,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.05,
        action_l2_penalty=-0.0005,
    )

    device = torch.device(cfg.device)
    env = UnityNavEnv(env_cfg)
    model = CNNGRUActorCritic(cfg.lidar_dim, cfg.low_dim, cfg.action_dim, gru_hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    writer.add_text("config", str(cfg))

    global_step = 0
    obs_np, _ = env.reset()
    obs_hist = init_obs_history(obs_np, cfg.seq_len)
    enhanced_obs = build_enhanced_obs(obs_hist, cfg)
    seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)

    episode_return = 0.0
    episode_len = 0
    train_ep_count = 0
    train_returns_window = deque(maxlen=50)
    train_lengths_window = deque(maxlen=50)
    train_success_window = deque(maxlen=50)
    train_collision_window = deque(maxlen=50)
    train_timeout_window = deque(maxlen=50)

    start_time = time.time()

    for update in range(1, cfg.total_updates + 1):
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
                train_ep_count += 1
                train_returns_window.append(float(episode_return))
                train_lengths_window.append(int(episode_len))
                train_success_window.append(float(info["success"]))
                train_collision_window.append(float(info["collision"]))
                train_timeout_window.append(float(info["timeout"]))

                writer.add_scalar("train/episode_return", float(episode_return), global_step)
                writer.add_scalar("train/episode_length", int(episode_len), global_step)
                writer.add_scalar("train/episode_success", float(info["success"]), global_step)
                writer.add_scalar("train/episode_collision", float(info["collision"]), global_step)
                writer.add_scalar("train/episode_timeout", float(info["timeout"]), global_step)
                writer.add_scalar("train/final_goal_dist", float(info["goal_dist"]), global_step)

                print(
                    f"[train ep] update={update:04d} step={global_step} "
                    f"ret={episode_return:.3f} len={episode_len} "
                    f"success={info['success']} collision={info['collision']} timeout={info['timeout']}"
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
            rewards=reward_buf,
            dones=done_buf,
            values=value_buf,
            next_value=next_value,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = cfg.rollout_steps
        batch_inds = np.arange(batch_size)
        last_pg_loss, last_v_loss, last_entropy, last_kl = 0.0, 0.0, 0.0, 0.0
        clipfracs = []

        for _ in range(cfg.update_epochs):
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
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(newvalue - mb_old_values, -cfg.clip_coef, cfg.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                last_pg_loss = float(pg_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())
                last_kl = float(approx_kl)

        sps = int(global_step / max(time.time() - start_time, 1e-6))
        writer.add_scalar("train/update", update, global_step)
        writer.add_scalar("train/global_step", global_step, global_step)
        writer.add_scalar("train/rollout_reward_mean", float(np.mean(rollout_rewards)), global_step)
        writer.add_scalar("train/loss_pi", last_pg_loss, global_step)
        writer.add_scalar("train/loss_v", last_v_loss, global_step)
        writer.add_scalar("train/entropy", last_entropy, global_step)
        writer.add_scalar("train/approx_kl", last_kl, global_step)
        writer.add_scalar("train/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("train/adv_mean", float(advantages.mean().item()), global_step)
        writer.add_scalar("train/return_mean", float(returns.mean().item()), global_step)
        writer.add_scalar("train/value_mean", float(value_buf.mean().item()), global_step)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/SPS", sps, global_step)
        writer.add_scalar("train/episodes_seen", train_ep_count, global_step)

        if train_returns_window:
            writer.add_scalar("train_window/return_mean_50", float(np.mean(train_returns_window)), global_step)
            writer.add_scalar("train_window/length_mean_50", float(np.mean(train_lengths_window)), global_step)
            writer.add_scalar("train_window/success_rate_50", float(np.mean(train_success_window)), global_step)
            writer.add_scalar("train_window/collision_rate_50", float(np.mean(train_collision_window)), global_step)
            writer.add_scalar("train_window/timeout_rate_50", float(np.mean(train_timeout_window)), global_step)

        print(
            f"update={update:04d} loss_pi={last_pg_loss:.4f} loss_v={last_v_loss:.4f} "
            f"entropy={last_entropy:.4f} kl={last_kl:.5f} sps={sps}"
        )

        if update % cfg.eval_every == 0:
            eval_stats = evaluate_policy(env, model, cfg, device, num_episodes=cfg.eval_episodes)
            writer.add_scalar("eval/return_mean", eval_stats["return_mean"], global_step)
            writer.add_scalar("eval/return_std", eval_stats["return_std"], global_step)
            writer.add_scalar("eval/length_mean", eval_stats["length_mean"], global_step)
            writer.add_scalar("eval/success_rate", eval_stats["success_rate"], global_step)
            writer.add_scalar("eval/collision_rate", eval_stats["collision_rate"], global_step)
            writer.add_scalar("eval/timeout_rate", eval_stats["timeout_rate"], global_step)
            writer.add_scalar("eval/final_goal_dist_mean", eval_stats["final_goal_dist_mean"], global_step)

            print(
                f"[eval] update={update:04d} ret={eval_stats['return_mean']:.3f}±{eval_stats['return_std']:.3f} "
                f"succ={eval_stats['success_rate']:.3f} coll={eval_stats['collision_rate']:.3f} "
                f"timeout={eval_stats['timeout_rate']:.3f} len={eval_stats['length_mean']:.1f}"
            )

            # 评估结束后恢复训练起点，避免把评估回合的状态串回训练轨迹
            obs_np, _ = env.reset()
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            enhanced_obs = build_enhanced_obs(obs_hist, cfg)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
            episode_return = 0.0
            episode_len = 0

        if update % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"ppo_gru_update_{update:04d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "env_cfg": env_cfg.__dict__,
                    "ppo_cfg": cfg.__dict__,
                    "model_type": "cnn_gru_ppo_tb",
                },
                save_path,
            )
            print(f"saved to {save_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()

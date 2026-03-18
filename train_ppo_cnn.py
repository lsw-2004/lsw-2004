import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from unity_env import UnityNavEnv, EnvConfig


# =========================
# Config
# =========================
@dataclass
class PPOConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    total_updates: int = 100
    rollout_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 2e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 10
    minibatch_size: int = 256

    obs_dim: int = 187
    lidar_dim: int = 180
    low_dim: int = 7
    action_dim: int = 2

    save_dir: str = "./checkpoints/cnn_ppo/first_test"
    save_every: int = 20


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

        # LiDAR 分支: (B, 180) -> (B, 1, 180)
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),   # -> (16, 90)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # -> (32, 45)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 23)
            nn.ReLU(),
            nn.Flatten(),
        )

        # 自动推导 lidar feature dim，避免手算错
        with torch.no_grad():
            dummy = torch.zeros(1, 1, lidar_dim)
            lidar_feat_dim = self.lidar_encoder(dummy).shape[1]

        self.lidar_fc = nn.Sequential(
            nn.Linear(lidar_feat_dim, 128),
            nn.Tanh(),
        )

        # 低维状态分支
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # 融合主干
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
        """
        obs: (B, 187)
        """
        lidar = obs[:, :180]         # (B, 180)
        low_dim = obs[:, 180:]       # (B, 7)

        lidar = lidar.unsqueeze(1)   # (B, 1, 180)
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

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        mean, logstd, value = self(obs)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, value.squeeze(-1)

    def get_value(self, obs: torch.Tensor):
        _, _, value = self(obs)
        return value.squeeze(-1)

    def get_deterministic_action(self, obs: torch.Tensor):
        mean, _, value = self(obs)
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


# =========================
# Train
# =========================
def main():
    env_cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\Builds\Project_1.exe",   # 改成你的真实路径
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=250,
        progress_gain=1.4,
        time_penalty=-0.0022,
        collision_penalty=-2.0,
        success_bonus=16.0,
        timeout_penalty=-2.2,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.045,
        action_l2_penalty=-0.0004,
    )

    cfg = PPOConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device(cfg.device)

    env = UnityNavEnv(env_cfg)
    model = CNNActorCritic(
        lidar_dim=cfg.lidar_dim,
        low_dim=cfg.low_dim,
        action_dim=cfg.action_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    global_step = 0
    obs_np, info = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

    episode_return = 0.0
    episode_len = 0

    for update in range(1, cfg.total_updates + 1):
        obs_buf = []
        action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        for step in range(cfg.rollout_steps):
            global_step += 1

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(obs.unsqueeze(0))
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
                value = value.squeeze(0)

            action_np = action.cpu().numpy()
            next_obs_np, reward, done, truncated, info = env.step(action_np)

            episode_return += reward
            episode_len += 1

            obs_buf.append(obs)
            action_buf.append(action)
            logprob_buf.append(logprob)
            reward_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))
            value_buf.append(value)

            if done:
                print(
                    f"[update {update:04d}] "
                    f"ep_ret={episode_return:.3f}, ep_len={episode_len}, "
                    f"success={info['success']}, collision={info['collision']}, timeout={info['timeout']}"
                )
                next_obs_np, _ = env.reset()
                episode_return = 0.0
                episode_len = 0

            obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = model.get_value(obs.unsqueeze(0)).squeeze(0)

        obs_buf = torch.stack(obs_buf)               # (T, 187)
        action_buf = torch.stack(action_buf)         # (T, 2)
        logprob_buf = torch.stack(logprob_buf)       # (T,)
        reward_buf = torch.stack(reward_buf)         # (T,)
        done_buf = torch.stack(done_buf)             # (T,)
        value_buf = torch.stack(value_buf)           # (T,)

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
        indices = np.arange(batch_size)

        last_pg_loss = 0.0
        last_v_loss = 0.0
        last_entropy = 0.0

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs_buf[mb_idx]
                mb_actions = action_buf[mb_idx]
                mb_old_logprob = logprob_buf[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = value_buf[mb_idx]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    mb_obs, mb_actions
                )

                logratio = newlogprob - mb_old_logprob
                ratio = torch.exp(logratio)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - cfg.clip_coef,
                    1.0 + cfg.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(
                    newvalue - mb_old_values,
                    -cfg.clip_coef,
                    cfg.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                last_pg_loss = pg_loss.item()
                last_v_loss = v_loss.item()
                last_entropy = entropy_loss.item()

        print(
            f"update={update:04d} "
            f"loss_pi={last_pg_loss:.4f} "
            f"loss_v={last_v_loss:.4f} "
            f"entropy={last_entropy:.4f}"
        )

        if update % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"ppo_update_{update:04d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "env_cfg": env_cfg.__dict__,
                    "ppo_cfg": cfg.__dict__,
                    "model_type": "cnn_ppo",
                },
                save_path,
            )
            print(f"saved to {save_path}")

    env.close()


if __name__ == "__main__":
    main()
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from unity_env import UnityNavEnv, EnvConfig


@dataclass
class PPOConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    total_updates: int = 400
    rollout_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 10
    minibatch_size: int = 256

    action_dim: int = 2
    obs_dim: int = 187

    save_dir: str = "./checkpoints/reward_shaping_20.0"
    save_every: int = 50


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

    def get_action_and_value(self, x, action=None):
        mean, logstd, value = self(x)
        std = torch.exp(logstd)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, value.squeeze(-1)

    def get_value(self, x):
        _, _, value = self(x)
        return value.squeeze(-1)


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

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


def main():
    env_cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\Builds\Project_1.exe",   # 改成你的 exe 路径
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=350,
        progress_gain=1.0,
        time_penalty=-0.005,
        collision_penalty=-2.0,
        success_bonus=20.0,
        timeout_penalty=-2.5,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.04,
        action_l2_penalty=-0.0003,
    )

    ppo_cfg = PPOConfig()
    os.makedirs(ppo_cfg.save_dir, exist_ok=True)

    device = torch.device(ppo_cfg.device)
    env = UnityNavEnv(env_cfg)
    model = ActorCritic(ppo_cfg.obs_dim, ppo_cfg.action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.lr)

    global_step = 0
    obs_np, info = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

    episode_return = 0.0
    episode_len = 0

    for update in range(1, ppo_cfg.total_updates + 1):
        obs_buf = []
        action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        for step in range(ppo_cfg.rollout_steps):
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

        obs_buf = torch.stack(obs_buf)
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
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = ppo_cfg.rollout_steps
        indices = np.arange(batch_size)

        for epoch in range(ppo_cfg.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, ppo_cfg.minibatch_size):
                end = start + ppo_cfg.minibatch_size
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
                    1.0 - ppo_cfg.clip_coef,
                    1.0 + ppo_cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(
                    newvalue - mb_old_values,
                    -ppo_cfg.clip_coef,
                    ppo_cfg.clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss + ppo_cfg.vf_coef * v_loss - ppo_cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

        print(
            f"update={update:04d} "
            f"loss_pi={pg_loss.item():.4f} "
            f"loss_v={v_loss.item():.4f} "
            f"entropy={entropy_loss.item():.4f}"
        )

        if update % ppo_cfg.save_every == 0:
            save_path = os.path.join(ppo_cfg.save_dir, f"ppo_update_{update:04d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "env_cfg": env_cfg.__dict__,
                    "ppo_cfg": ppo_cfg.__dict__,
                },
                save_path,
            )
            print(f"saved to {save_path}")

    env.close()


if __name__ == "__main__":
    main()
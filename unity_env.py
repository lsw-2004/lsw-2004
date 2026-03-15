import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


@dataclass
class EnvConfig:
    file_name: Optional[str] = None          # Unity build 路径；编辑器模式可设为 None
    behavior_name: str = "Navtest?team=0"  # Unity 场景中 Agent 的 behavior name
    seed: int = 1
    no_graphics: bool = False
    timeout_wait: int = 60

    obs_size: int = 187
    lidar_dim: int = 180

    reach_goal_radius: float = 0.5
    max_steps: int = 300

    # reward 参数
    progress_gain: float = 1.0
    time_penalty: float = -0.002
    collision_penalty: float = -1.5
    success_bonus: float = 2.0
    timeout_penalty: float = -0.5
    near_obstacle_threshold: float = 0.3
    near_obstacle_penalty: float = -0.02
    action_l2_penalty: float = -0.001


class UnityNavEnv:
    """
    把 Unity ML-Agents build 包成一个单智能体连续控制环境。
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

        self.env = UnityEnvironment(
            file_name=cfg.file_name,
            seed=cfg.seed,
            no_graphics=cfg.no_graphics,
            timeout_wait=cfg.timeout_wait,
        )
        self.env.reset()

        self.behavior_name = cfg.behavior_name
        if self.behavior_name not in self.env.behavior_specs:
            names = list(self.env.behavior_specs.keys())
            raise ValueError(
                f"找不到 behavior_name={cfg.behavior_name}，当前可用行为有: {names}"
            )

        self.spec = self.env.behavior_specs[self.behavior_name]

        # 当前假设：单智能体
        self.agent_id = None
        self.last_obs = None
        self.last_dist = None
        self.step_count = 0

        self._bind_first_agent()

    def _bind_first_agent(self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(decision_steps) == 0 and len(terminal_steps) == 0:
            # 有时 reset 后需要先空推进一帧
            self.env.step()
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        if len(decision_steps) > 0:
            self.agent_id = list(decision_steps.agent_id)[0]
            obs = decision_steps[self.agent_id].obs[0]
        elif len(terminal_steps) > 0:
            self.agent_id = list(terminal_steps.agent_id)[0]
            obs = terminal_steps[self.agent_id].obs[0]
        else:
            raise RuntimeError("没有检测到任何 agent，请检查 Unity 场景是否正常启动。")

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._check_obs(obs)

        self.last_obs = obs
        self.last_dist = self._extract_goal_dist(obs)
        self.step_count = 0

    def _check_obs(self, obs: np.ndarray):
        if obs.shape[0] != self.cfg.obs_size:
            raise ValueError(
                f"观测维度不匹配，期望 {self.cfg.obs_size}，实际 {obs.shape[0]}"
            )

    def _extract_goal_dist(self, obs: np.ndarray) -> float:
        # 结构:
        # [0:180] lidar
        # [180] dir.x
        # [181] dir.z
        # [182] normalized distance
        # [183] angle
        # [184] linear vel
        # [185] angular vel
        # [186] collision flag
        norm_dist = float(obs[182])
        # Unity 里是 dist / _maxScenarioSize，当前 _maxScenarioSize = 30f
        return norm_dist * 30.0

    def _extract_min_lidar(self, obs: np.ndarray) -> float:
        lidar = obs[:self.cfg.lidar_dim]
        # Unity 里 lidar 已经除以 maxDistance；默认 maxDistance=10f
        return float(np.min(lidar) * 10.0)

    def _extract_collision(self, obs: np.ndarray) -> bool:
        return bool(obs[186] > 0.5)

    def reset(self) -> Tuple[np.ndarray, Dict]:
        # 关键点：
        # 你当前 Unity 中 reset 走的是 Agent episode begin + EpisodeManager.ResetEpisode()
        # 对于外部控制，最稳妥方式是直接 reset 整个环境。
        self.env.reset()
        self._bind_first_agent()

        info = {
            "goal_dist": self.last_dist,
            "step_count": self.step_count,
        }
        return self.last_obs.copy(), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(1, 2)
        action = np.clip(action, -1.0, 1.0)

        action_tuple = ActionTuple(continuous=action)
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        done = False
        truncated = False

        if self.agent_id in terminal_steps:
            step_result = terminal_steps[self.agent_id]
            obs = np.asarray(step_result.obs[0], dtype=np.float32).reshape(-1)
            done = True
        elif self.agent_id in decision_steps:
            step_result = decision_steps[self.agent_id]
            obs = np.asarray(step_result.obs[0], dtype=np.float32).reshape(-1)
        else:
            # 极少数情况下 agent id 变化，重新绑定
            self._bind_first_agent()
            obs = self.last_obs.copy()

        self._check_obs(obs)

        self.step_count += 1

        dist_now = self._extract_goal_dist(obs)
        min_lidar = self._extract_min_lidar(obs)
        collision = self._extract_collision(obs)

        success = dist_now < self.cfg.reach_goal_radius
        timeout = self.step_count >= self.cfg.max_steps

        reward = 0.0

        # 1) 进度奖励
        progress = self.last_dist - dist_now
        reward += self.cfg.progress_gain * progress

        # 2) 时间惩罚
        reward += self.cfg.time_penalty

        # 3) 近障碍惩罚
        if min_lidar < self.cfg.near_obstacle_threshold:
            reward += self.cfg.near_obstacle_penalty

        # 4) 动作 L2 惩罚
        reward += self.cfg.action_l2_penalty * float(np.sum(np.square(action)))

        # 5) 终止奖励/惩罚
        if collision:
            reward += self.cfg.collision_penalty
            done = True
            timeout = False

        elif success:
            reward += self.cfg.success_bonus
            done = True
            collision = False

        elif timeout:
            reward += self.cfg.timeout_penalty
            done = True
            truncated = True
        info = {
            "goal_dist": dist_now,
            "min_lidar": min_lidar,
            "collision": collision,
            "success": success,
            "timeout": timeout,
            "step_count": self.step_count,
        }

        self.last_obs = obs
        self.last_dist = dist_now

        return obs.copy(), float(reward), bool(done), bool(truncated), info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


if __name__ == "__main__":
    cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\Builds\Project_1.exe",   # 改成你的 build 路径
        behavior_name="Navtest?team=0",
        no_graphics=False,
    )

    env = UnityNavEnv(cfg)

    obs, info = env.reset()
    print("reset obs shape:", obs.shape, info)

    for i in range(20):
        action = np.array([0.5, 0.0], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)
        print(i, reward, done, truncated, info)
        if done:
            obs, info = env.reset()
            print("episode reset", info)

    env.close()
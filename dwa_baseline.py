import math
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np

from unity_env import UnityNavEnv, EnvConfig


@dataclass
class DWAConfig:
    # 机器人控制上限（要与你 Unity 中 RobotController 的速度上限一致或同量纲）
    v_max: float = 1.0          # 对应动作归一化后的上限
    w_max: float = 1.0          # 对应动作归一化后的上限

    # 采样数
    n_v: int = 7
    n_w: int = 11

    # 预测时域
    horizon_steps: int = 8
    dt: float = 0.1

    # 评分权重
    heading_weight: float = 1.2
    clearance_weight: float = 1.5
    velocity_weight: float = 0.2
    goal_weight: float = 1.0

    # 安全参数
    robot_radius: float = 0.25
    brake_margin: float = 0.05
    hard_clearance: float = 0.18

    # 转向平滑
    max_delta_v: float = 0.25
    max_delta_w: float = 0.35


class DWAPolicy:
    """
    基于当前 187 维观测直接做 DWA。
    观测结构:
      0:180    lidar (已归一化到[0,1]，unity_env里还原为米更方便)
      180      goal_dir_x
      181      goal_dir_z
      182      goal_dist_norm
      183      goal_angle_norm (-1~1)
      184      linear_vel_norm
      185      angular_vel_norm
      186      collision_flag
    """
    def __init__(self, cfg: DWAConfig):
        self.cfg = cfg
        self.prev_action = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.prev_action[:] = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        lidar_norm = obs[:180]
        goal_dir = obs[180:182]
        goal_dist = obs[182] * 30.0
        goal_angle = obs[183] * math.pi
        cur_v = float(obs[184])   # 已归一化
        cur_w = float(obs[185])   # 已归一化

        lidar_m = lidar_norm * 10.0

        v_candidates = np.linspace(
            max(-self.cfg.v_max, cur_v - self.cfg.max_delta_v),
            min(self.cfg.v_max, cur_v + self.cfg.max_delta_v),
            self.cfg.n_v
        )
        w_candidates = np.linspace(
            max(-self.cfg.w_max, cur_w - self.cfg.max_delta_w),
            min(self.cfg.w_max, cur_w + self.cfg.max_delta_w),
            self.cfg.n_w
        )

        best_score = -1e9
        best_action = np.array([0.0, 0.0], dtype=np.float32)

        for v in v_candidates:
            for w in w_candidates:
                score = self._score_action(v, w, goal_angle, goal_dist, lidar_m)
                if score > best_score:
                    best_score = score
                    best_action[:] = [v, w]

        self.prev_action = best_action.copy()
        return best_action

    def _score_action(
        self,
        v: float,
        w: float,
        goal_angle: float,
        goal_dist: float,
        lidar_m: np.ndarray
    ) -> float:
        # 1) heading score：预测末端朝向和目标夹角越小越好
        pred_heading = w * self.cfg.horizon_steps * self.cfg.dt
        heading_err = abs(self._wrap_to_pi(goal_angle - pred_heading))
        heading_score = 1.0 - min(heading_err / math.pi, 1.0)

        # 2) clearance score：沿当前(v,w)大致前进方向，看激光余量
        clearance = self._estimate_clearance(v, w, lidar_m)
        if clearance < self.cfg.hard_clearance:
            return -1e6  # 直接判不可行

        clearance_score = np.tanh(clearance)

        # 3) velocity score：鼓励前进
        velocity_score = max(v, 0.0)

        # 4) goal score：离目标远时鼓励更前进，近时更强调转正
        goal_score = np.tanh(goal_dist / 5.0)

        score = (
            self.cfg.heading_weight * heading_score +
            self.cfg.clearance_weight * clearance_score +
            self.cfg.velocity_weight * velocity_score +
            self.cfg.goal_weight * goal_score
        )

        # 对过大角速度做一点轻微惩罚
        score -= 0.05 * abs(w)
        return float(score)

    def _estimate_clearance(self, v: float, w: float, lidar_m: np.ndarray) -> float:
        """
        用一个近似方法：
        - 根据当前转向趋势选取前方扇区
        - 估计该方向上的最近障碍余量
        """
        center_deg = 90  # 假设前方对应中间
        steer_offset = int(np.clip(w, -1.0, 1.0) * 35)
        center = center_deg + steer_offset

        half_width = 12 if abs(w) < 0.3 else 18
        left = max(0, center - half_width)
        right = min(179, center + half_width)

        sector = lidar_m[left:right + 1]
        if len(sector) == 0:
            return 0.0

        raw_clearance = float(np.min(sector))
        clearance = raw_clearance - self.cfg.robot_radius - self.cfg.brake_margin

        # 若线速度大，给更多安全余量要求
        clearance -= max(v, 0.0) * 0.15
        return clearance

    @staticmethod
    def _wrap_to_pi(x: float) -> float:
        return (x + math.pi) % (2 * math.pi) - math.pi


def run_dwa_episode(env: UnityNavEnv, policy: DWAPolicy, render: bool = False):
    obs, info = env.reset()
    policy.reset()

    done = False
    truncated = False
    ep_ret = 0.0
    ep_len = 0
    last_info = {}

    while not done:
        action = policy.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        ep_ret += reward
        ep_len += 1
        last_info = info

    result = {
        "return": ep_ret,
        "length": ep_len,
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "timeout": bool(last_info.get("timeout", False)),
        "final_goal_dist": float(last_info.get("goal_dist", np.nan)),
    }
    return result


if __name__ == "__main__":
    env_cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\YourBuild\NavEnv.exe",   # 改成你的路径
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=200,
        progress_gain=2.0,
        time_penalty=-0.001,
        collision_penalty=-1.5,
        success_bonus=5.0,
        timeout_penalty=-1.0,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.03,
        action_l2_penalty=-0.0005,
    )

    policy_cfg = DWAConfig()
    env = UnityNavEnv(env_cfg)
    policy = DWAPolicy(policy_cfg)

    n_episodes = 20
    results = []
    for i in range(n_episodes):
        result = run_dwa_episode(env, policy)
        results.append(result)
        print(f"[DWA][{i+1:03d}] {result}")

    env.close()

    success_rate = np.mean([r["success"] for r in results])
    collision_rate = np.mean([r["collision"] for r in results])
    timeout_rate = np.mean([r["timeout"] for r in results])
    avg_len = np.mean([r["length"] for r in results])
    avg_ret = np.mean([r["return"] for r in results])

    print("\n=== DWA Summary ===")
    print(f"success_rate   : {success_rate:.3f}")
    print(f"collision_rate : {collision_rate:.3f}")
    print(f"timeout_rate   : {timeout_rate:.3f}")
    print(f"avg_len        : {avg_len:.2f}")
    print(f"avg_return     : {avg_ret:.3f}")
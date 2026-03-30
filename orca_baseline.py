"""
DWA (Dynamic Window Approach) Baseline - 360° LiDAR + 近目标增强版

目标：
1. 保持整体 DWA 逻辑不变
2. 保留 balanced 版本较低的 collision
3. 增强“最后 1 米”到达目标能力
4. 将 success 判定半径改为 0.6
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from unity_env import UnityNavEnv, EnvConfig


# =============================================================================
# DWA Config
# =============================================================================
@dataclass
class DWAConfig:
    """DWA 算法配置 - 360° LiDAR + 近目标增强版"""

    # LiDAR 参数
    lidar_dim: int = 360
    lidar_max_distance: float = 10.0

    # 机器人运动学参数
    max_vel_x: float = 1.0
    max_vel_theta: float = 1.5
    min_vel_x: float = 0.0
    min_vel_theta: float = -1.5

    # 加速度限制
    acc_lim_x: float = 2.5
    acc_lim_theta: float = 3.2

    # 速度采样
    vx_samples: int = 12
    vtheta_samples: int = 20

    # 轨迹模拟参数
    sim_time: float = 1.4
    sim_granularity: float = 0.05

    # 评分权重
    path_distance_bias: float = 35.0
    goal_distance_bias: float = 22.0
    occdist_scale: float = 2.6
    oscillation_scale: float = 0.22

    # 推进权重
    progress_bonus_weight: float = 4.6
    low_speed_penalty_weight: float = 0.6
    spin_penalty_weight: float = 1.1

    # 障碍物参数
    robot_radius: float = 0.25
    inflation_radius: float = 0.6
    safety_margin: float = 0.1
    min_safe_distance: float = 0.38

    # 动态障碍物应对
    dynamic_safety_factor: float = 1.5

    # 前向预测
    forward_point_distance: float = 0.325

    # 速度平滑
    vel_smooth_factor: float = 0.035

    # stuck / recovery
    stuck_velocity_threshold: float = 0.08
    stuck_progress_threshold: float = 0.02
    stuck_window: int = 40
    recovery_steps: int = 6
    recovery_vx: float = 0.18
    recovery_vw_base: float = 0.48

    # 近目标增强
    near_goal_dist: float = 1.2
    near_goal_turn_only_angle: float = 0.35   # rad，角度偏差较大时先转向
    near_goal_slow_vx: float = 0.22           # m/s
    near_goal_fast_vx: float = 0.35           # m/s
    near_goal_max_w: float = 0.75             # rad/s
    near_goal_front_safe: float = 0.55        # m
    near_goal_emergency_front: float = 0.35   # m


# =============================================================================
# DWA Policy
# =============================================================================
class DWAPolicy:
    """
    DWA 策略实现 - 360° LiDAR + 近目标增强版
    保持主逻辑不变：
    - 动态窗口采样
    - 轨迹模拟
    - 安全性检查
    - 评分选优
    - stuck -> recovery
    - 新增 near-goal mode
    """

    def __init__(self, cfg: DWAConfig):
        self.cfg = cfg
        self.prev_vx = 0.0
        self.prev_vtheta = 0.0
        self.prev_actions = []

        self.velocity_history = []
        self.goal_dist_history = []
        self.is_recovery_mode = False
        self.recovery_counter = 0
        self.recovery_direction = 0.0
        self.total_steps = 0
        self.consecutive_low_vel = 0

        self._emergency_avoid_called = False
        self._near_goal_mode_called = False

    def reset(self):
        self.prev_vx = 0.0
        self.prev_vtheta = 0.0
        self.prev_actions = []
        self.velocity_history = []
        self.goal_dist_history = []
        self.is_recovery_mode = False
        self.recovery_counter = 0
        self.recovery_direction = 0.0
        self.total_steps = 0
        self.consecutive_low_vel = 0
        self._emergency_avoid_called = False
        self._near_goal_mode_called = False

    # -------------------------------------------------------------------------
    # 360° LiDAR helpers
    # -------------------------------------------------------------------------
    def _index_to_angle_deg(self, idx: int) -> float:
        if self.cfg.lidar_dim <= 1:
            return 0.0
        return -180.0 + 360.0 * idx / (self.cfg.lidar_dim - 1)

    def _angle_to_lidar_idx(self, angle_deg: float) -> int:
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0

        idx = int(round((angle_deg + 180.0) / 360.0 * (self.cfg.lidar_dim - 1)))
        return max(0, min(self.cfg.lidar_dim - 1, idx))

    def _sector_min(self, lidar_m: np.ndarray, center_deg: float, half_width_deg: float) -> float:
        vals = []
        for i in range(self.cfg.lidar_dim):
            ang = self._index_to_angle_deg(i)
            diff = ((ang - center_deg + 180.0) % 360.0) - 180.0
            if abs(diff) <= half_width_deg:
                vals.append(lidar_m[i])

        if len(vals) == 0:
            return float(np.min(lidar_m))
        return float(np.min(vals))

    def _get_front_left_right(self, lidar_m: np.ndarray) -> Tuple[float, float, float]:
        front_min = self._sector_min(lidar_m, center_deg=0.0, half_width_deg=18.0)
        left_min = self._sector_min(lidar_m, center_deg=-90.0, half_width_deg=35.0)
        right_min = self._sector_min(lidar_m, center_deg=90.0, half_width_deg=35.0)
        return front_min, left_min, right_min

    # -------------------------------------------------------------------------
    # near-goal helper
    # -------------------------------------------------------------------------
    def _near_goal_control(self, lidar_m: np.ndarray, goal_dist: float, goal_angle: float) -> np.ndarray:
        """
        近目标特判：
        - 优先把朝向对准目标
        - 若前方安全，则小速度靠近
        - 若前方太近，则先转向脱离局部卡住
        """
        self._near_goal_mode_called = True

        front_min = self._sector_min(lidar_m, center_deg=0.0, half_width_deg=20.0)
        left_front_min = self._sector_min(lidar_m, center_deg=-35.0, half_width_deg=20.0)
        right_front_min = self._sector_min(lidar_m, center_deg=35.0, half_width_deg=20.0)

        # 前方非常危险，优先原地转向到更空的一侧
        if front_min < self.cfg.near_goal_emergency_front:
            action_v = 0.0
            if left_front_min >= right_front_min:
                action_w = self.cfg.near_goal_max_w
            else:
                action_w = -self.cfg.near_goal_max_w
            return np.array(
                [action_v / self.cfg.max_vel_x, action_w / self.cfg.max_vel_theta],
                dtype=np.float32
            )

        # 目标角偏差较大时，先转向对准
        if abs(goal_angle) > self.cfg.near_goal_turn_only_angle:
            action_v = 0.0
            action_w = np.clip(1.3 * goal_angle, -self.cfg.near_goal_max_w, self.cfg.near_goal_max_w)
            return np.array(
                [action_v / self.cfg.max_vel_x, action_w / self.cfg.max_vel_theta],
                dtype=np.float32
            )

        # 目标已经比较对准，且前方安全，直接慢速靠近
        if front_min > self.cfg.near_goal_front_safe:
            if goal_dist > 0.8:
                action_v = self.cfg.near_goal_fast_vx
            else:
                action_v = self.cfg.near_goal_slow_vx
            action_w = np.clip(1.0 * goal_angle, -0.40, 0.40)
        else:
            # 前方一般，不完全停住，慢速+微调角度
            action_v = 0.12
            if left_front_min > right_front_min + 0.05:
                steer_bias = 0.18
            elif right_front_min > left_front_min + 0.05:
                steer_bias = -0.18
            else:
                steer_bias = 0.0
            action_w = np.clip(0.9 * goal_angle + steer_bias, -0.50, 0.50)

        return np.array(
            [action_v / self.cfg.max_vel_x, action_w / self.cfg.max_vel_theta],
            dtype=np.float32
        )

    # -------------------------------------------------------------------------
    # Main action
    # -------------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> np.ndarray:
        self.total_steps += 1
        self._emergency_avoid_called = False
        self._near_goal_mode_called = False

        lidar_norm = obs[:self.cfg.lidar_dim]
        goal_dist = float(obs[self.cfg.lidar_dim + 2]) * 30.0
        goal_angle = float(obs[self.cfg.lidar_dim + 3]) * math.pi
        cur_v = float(obs[self.cfg.lidar_dim + 4]) * self.cfg.max_vel_x
        cur_w = float(obs[self.cfg.lidar_dim + 5]) * self.cfg.max_vel_theta

        lidar_m = lidar_norm * self.cfg.lidar_max_distance

        # 近目标模式优先
        if goal_dist < self.cfg.near_goal_dist:
            # 近目标时不让 recovery 接管，避免最后 1 米还在恢复模式里磨蹭
            self.is_recovery_mode = False
            self.recovery_counter = 0
            return self._near_goal_control(lidar_m, goal_dist, goal_angle)

        self.velocity_history.append(cur_v)
        self.goal_dist_history.append(goal_dist)
        if len(self.velocity_history) > self.cfg.stuck_window:
            self.velocity_history.pop(0)
            self.goal_dist_history.pop(0)

        is_stuck = self._detect_stuck(lidar_m)

        if self.is_recovery_mode:
            self.recovery_counter += 1
            if self.recovery_counter >= self.cfg.recovery_steps:
                self.is_recovery_mode = False
                self.recovery_counter = 0
                self.velocity_history = []
                self.goal_dist_history = []
                self.consecutive_low_vel = 0
            else:
                return self._execute_recovery(lidar_m, goal_angle)

        if is_stuck and not self.is_recovery_mode:
            self.is_recovery_mode = True
            self.recovery_counter = 0
            left_min = self._sector_min(lidar_m, center_deg=-90.0, half_width_deg=60.0)
            right_min = self._sector_min(lidar_m, center_deg=90.0, half_width_deg=60.0)
            self.recovery_direction = 1.0 if left_min > right_min else -1.0
            return self._execute_recovery(lidar_m, goal_angle)

        v_min, v_max, w_min, w_max = self._compute_dynamic_window(cur_v, cur_w)

        vx_samples = np.linspace(v_min, v_max, self.cfg.vx_samples)
        vtheta_samples = np.linspace(w_min, w_max, self.cfg.vtheta_samples)

        best_score = -float("inf")
        best_vx = 0.0
        best_vtheta = 0.0
        safe_candidates = []

        for vx in vx_samples:
            for vtheta in vtheta_samples:
                trajectory = self._simulate_trajectory(vx, vtheta)

                is_safe, min_dist = self._check_trajectory_safety(trajectory, lidar_m)
                if not is_safe:
                    continue

                score = self._score_trajectory(
                    trajectory, vx, vtheta,
                    goal_dist, goal_angle, lidar_m, min_dist
                )

                if vx > 0.15:
                    score += 0.25
                if vx > 0.30 and abs(vtheta) < 0.45:
                    score += 0.15

                safe_candidates.append((score, vx, vtheta, min_dist))

                if score > best_score:
                    best_score = score
                    best_vx = vx
                    best_vtheta = vtheta

        if not safe_candidates:
            return self._emergency_avoid(lidar_m, goal_angle)

        self.prev_vx = best_vx
        self.prev_vtheta = best_vtheta

        if best_vx < 0.05:
            self.consecutive_low_vel += 1
        else:
            self.consecutive_low_vel = 0

        self.prev_actions.append((best_vx, best_vtheta))
        if len(self.prev_actions) > 10:
            self.prev_actions.pop(0)

        action_v = best_vx / self.cfg.max_vel_x
        action_w = best_vtheta / self.cfg.max_vel_theta
        return np.array([action_v, action_w], dtype=np.float32)

    def _detect_stuck(self, lidar_m: np.ndarray) -> bool:
        if len(self.velocity_history) < self.cfg.stuck_window:
            return False

        front_min, left_min, right_min = self._get_front_left_right(lidar_m)

        if len(self.goal_dist_history) >= self.cfg.stuck_window:
            progress = self.goal_dist_history[0] - self.goal_dist_history[-1]

            if front_min > 0.9 or max(left_min, right_min) > 1.2:
                progress_threshold = 0.12
            else:
                progress_threshold = 0.18

            if progress < progress_threshold and np.mean(self.velocity_history) < 0.20:
                return True

        if self.consecutive_low_vel > 40 and front_min < 0.8:
            return True

        return False

    def _execute_recovery(self, lidar_m: np.ndarray, goal_angle: float) -> np.ndarray:
        front_min, left_min, right_min = self._get_front_left_right(lidar_m)
        safe_dist = self.cfg.robot_radius + self.cfg.safety_margin + 0.2

        if front_min > 1.2:
            action_v = self.cfg.recovery_vx
            action_w = 0.0

        elif front_min > safe_dist:
            action_v = self.cfg.recovery_vx * 0.75
            if abs(goal_angle) < math.pi / 4:
                action_w = np.sign(goal_angle) * 0.12
            else:
                action_w = 0.0

        else:
            if left_min > right_min + 0.1:
                action_w = self.cfg.recovery_vw_base
            elif right_min > left_min + 0.1:
                action_w = -self.cfg.recovery_vw_base
            else:
                action_w = self.cfg.recovery_vw_base * self.recovery_direction

            side_min = max(left_min, right_min)
            if side_min > safe_dist * 1.45:
                action_v = self.cfg.recovery_vx * 0.25
            else:
                action_v = 0.0

        return np.array([
            action_v / self.cfg.max_vel_x,
            action_w / self.cfg.max_vel_theta
        ], dtype=np.float32)

    def _emergency_avoid(self, lidar_m: np.ndarray, goal_angle: float) -> np.ndarray:
        self._emergency_avoid_called = True

        front_min, left_min, right_min = self._get_front_left_right(lidar_m)
        safe_dist = self.cfg.robot_radius + self.cfg.safety_margin

        if front_min > safe_dist * 2.2:
            action_v = 0.28
            action_w = 0.0
        elif left_min > right_min and left_min > safe_dist:
            action_w = 0.55
            action_v = 0.0
        elif right_min > safe_dist:
            action_w = -0.55
            action_v = 0.0
        else:
            action_v = 0.0
            action_w = 0.58 if left_min >= right_min else -0.58

        return np.array([action_v, action_w], dtype=np.float32)

    def _compute_dynamic_window(self, cur_v: float, cur_w: float) -> Tuple[float, float, float, float]:
        dt = 0.1

        v_min_accel = cur_v - self.cfg.acc_lim_x * dt
        v_max_accel = cur_v + self.cfg.acc_lim_x * dt
        w_min_accel = cur_w - self.cfg.acc_lim_theta * dt
        w_max_accel = cur_w + self.cfg.acc_lim_theta * dt

        v_min = max(self.cfg.min_vel_x, v_min_accel)
        v_max = min(self.cfg.max_vel_x, v_max_accel)
        w_min = max(self.cfg.min_vel_theta, w_min_accel)
        w_max = min(self.cfg.max_vel_theta, w_max_accel)

        return v_min, v_max, w_min, w_max

    def _simulate_trajectory(self, vx: float, vtheta: float) -> List[Tuple[float, float, float]]:
        trajectory = []
        x, y, theta = 0.0, 0.0, 0.0

        n_steps = int(self.cfg.sim_time / self.cfg.sim_granularity)
        for _ in range(n_steps):
            x += vx * self.cfg.sim_granularity * math.sin(theta)
            y += vx * self.cfg.sim_granularity * math.cos(theta)
            theta += vtheta * self.cfg.sim_granularity
            trajectory.append((x, y, theta))

        return trajectory

    def _check_trajectory_safety(
        self,
        trajectory: List[Tuple[float, float, float]],
        lidar_m: np.ndarray
    ) -> Tuple[bool, float]:
        min_dist = float("inf")
        base_required_dist = self.cfg.robot_radius + self.cfg.safety_margin

        for i, (x, y, theta) in enumerate(trajectory):
            point_angle_deg = math.degrees(math.atan2(x, y + 1e-6))
            point_dist = math.sqrt(x**2 + y**2)

            lidar_idx = self._angle_to_lidar_idx(point_angle_deg)

            if abs(point_angle_deg) < 30.0:
                idx_range = max(1, int(round(5 * self.cfg.lidar_dim / 180.0)))
            else:
                idx_range = max(1, int(round(2.5 * self.cfg.lidar_dim / 180.0)))

            idx_start = max(0, lidar_idx - idx_range)
            idx_end = min(self.cfg.lidar_dim, lidar_idx + idx_range + 1)
            lidar_dist = float(np.min(lidar_m[idx_start:idx_end]))

            dist_to_obs = lidar_dist - point_dist
            min_dist = min(min_dist, dist_to_obs)

            time_factor = 1.0 + (i / max(1, len(trajectory))) * 0.24
            required_dist = base_required_dist * time_factor

            if dist_to_obs < required_dist:
                return False, min_dist

        return True, min_dist

    def _score_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]],
        vx: float,
        vtheta: float,
        goal_dist: float,
        goal_angle: float,
        lidar_m: np.ndarray,
        min_dist: float
    ) -> float:
        if not trajectory:
            return -float("inf")

        end_x, end_y, end_theta = trajectory[-1]

        # 1. 目标距离得分
        goal_x = goal_dist * math.sin(goal_angle)
        goal_y = goal_dist * math.cos(goal_angle)
        dist_to_goal = math.sqrt((end_x - goal_x) ** 2 + (end_y - goal_y) ** 2)
        path_score = self.cfg.path_distance_bias * (goal_dist - dist_to_goal)

        # 2. 目标方向得分
        angle_to_goal = math.atan2(goal_x - end_x, goal_y - end_y)
        heading_diff = abs(self._normalize_angle(angle_to_goal - end_theta))
        goal_score = self.cfg.goal_distance_bias * (math.pi - heading_diff)

        # 3. 障碍物代价
        safe_dist = self.cfg.robot_radius + self.cfg.safety_margin + self.cfg.min_safe_distance
        front_min = self._sector_min(lidar_m, center_deg=0.0, half_width_deg=18.0)

        occ_cost = 0.0
        if min_dist < safe_dist:
            dist_ratio = (safe_dist - min_dist) / max(1e-6, safe_dist)
            occ_cost += self.cfg.occdist_scale * (dist_ratio ** 1.7) * 68.0

        front_safe = self.cfg.robot_radius + self.cfg.safety_margin + 0.22
        if front_min < front_safe and vx > 0.30:
            front_penalty = (
                self.cfg.occdist_scale * 1.8 *
                (front_safe - front_min) / max(1e-6, front_safe) * vx
            )
            occ_cost += front_penalty

        # 4. 前进奖励
        forward_progress = end_y
        progress_bonus = self.cfg.progress_bonus_weight * max(0.0, forward_progress)

        # 5. 速度得分：较克制
        if front_min > 1.4:
            vel_score = vx * 1.25
        elif front_min > 1.0:
            vel_score = vx * 0.85
        elif front_min > 0.7:
            vel_score = vx * 0.40
        else:
            vel_score = vx * 0.10

        # 6. 低速惩罚
        low_speed_penalty = 0.0
        if vx < 0.08 and min_dist > safe_dist * 0.80:
            low_speed_penalty = self.cfg.low_speed_penalty_weight * (0.08 - vx) / 0.08

        # 7. 原地转圈惩罚
        spin_penalty = 0.0
        if vx < 0.08 and abs(vtheta) > 0.45:
            spin_penalty = self.cfg.spin_penalty_weight * (abs(vtheta) / self.cfg.max_vel_theta)

        # 8. 平滑惩罚
        smooth_penalty = self.cfg.vel_smooth_factor * abs(vtheta - self.prev_vtheta)

        # 9. 震荡惩罚
        oscillation_penalty = 0.0
        if len(self.prev_actions) >= 4:
            recent = self.prev_actions[-4:]
            if (recent[0][1] * recent[2][1] < 0 and recent[1][1] * recent[3][1] < 0):
                oscillation_penalty = self.cfg.oscillation_scale * abs(vtheta)

        total_score = (
            path_score + goal_score + vel_score + progress_bonus
            - occ_cost - low_speed_penalty - spin_penalty
            - smooth_penalty - oscillation_penalty
        )
        return total_score

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# =============================================================================
# Evaluation
# =============================================================================
def run_dwa_episode(env: UnityNavEnv, policy: DWAPolicy) -> Dict:
    obs, info = env.reset()
    policy.reset()

    done = False
    ep_ret = 0.0
    ep_len = 0
    last_info = {}

    min_lidar_list = []
    velocity_list = []
    recovery_count = 0
    emergency_count = 0
    near_goal_count = 0

    while not done:
        action = policy.act(obs)
        obs, reward, done, truncated, info = env.step(action)

        ep_ret += reward
        ep_len += 1
        last_info = info

        min_lidar_list.append(float(np.min(obs[:policy.cfg.lidar_dim]) * policy.cfg.lidar_max_distance))
        velocity_list.append(float(obs[policy.cfg.lidar_dim + 4]))

        if policy.is_recovery_mode:
            recovery_count += 1
        if policy._emergency_avoid_called:
            emergency_count += 1
        if policy._near_goal_mode_called:
            near_goal_count += 1

    return {
        "return": ep_ret,
        "length": ep_len,
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "timeout": bool(last_info.get("timeout", False)),
        "final_goal_dist": float(last_info.get("goal_dist", np.nan)),
        "min_lidar_mean": float(np.mean(min_lidar_list)) if min_lidar_list else 0.0,
        "min_lidar_min": float(np.min(min_lidar_list)) if min_lidar_list else 0.0,
        "velocity_mean": float(np.mean(velocity_list)) if velocity_list else 0.0,
        "recovery_steps": recovery_count,
        "emergency_steps": emergency_count,
        "near_goal_steps": near_goal_count,
    }


def evaluate_dwa(env_cfg: EnvConfig, policy_cfg: DWAConfig, n_episodes: int = 50) -> Dict:
    env = UnityNavEnv(env_cfg)
    policy = DWAPolicy(policy_cfg)

    results = []
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print("DWA Baseline Evaluation (360° LiDAR, near-goal enhanced)")
    print(f"{'=' * 60}")
    print(f"Episodes: {n_episodes}")
    print(f"Environment: {env_cfg.file_name}")
    print("Config:")
    print(f"  - lidar_dim: {policy_cfg.lidar_dim}")
    print(f"  - robot_radius: {policy_cfg.robot_radius}")
    print(f"  - safety_margin: {policy_cfg.safety_margin}")
    print(f"  - min_safe_distance: {policy_cfg.min_safe_distance}")
    print(f"  - occdist_scale: {policy_cfg.occdist_scale}")
    print(f"  - stuck_window: {policy_cfg.stuck_window}")
    print(f"  - recovery_steps: {policy_cfg.recovery_steps}")
    print(f"  - near_goal_dist: {policy_cfg.near_goal_dist}")
    print(f"  - vx_samples: {policy_cfg.vx_samples}")
    print(f"  - vtheta_samples: {policy_cfg.vtheta_samples}")
    print(f"{'=' * 60}\n")

    for i in range(n_episodes):
        result = run_dwa_episode(env, policy)
        results.append(result)

        status = "SUCCESS" if result["success"] else ("COLLISION" if result["collision"] else "TIMEOUT")
        print(
            f"[DWA][{i + 1:03d}] len={result['length']:3d} | "
            f"dist={result['final_goal_dist']:.2f}m | "
            f"vel={result['velocity_mean']:.3f} | "
            f"rec={result['recovery_steps']:3d} | "
            f"ng={result['near_goal_steps']:3d} | "
            f"{status}"
        )

    env.close()
    eval_time = time.time() - start_time

    success_rate = np.mean([r["success"] for r in results]) if results else 0.0
    collision_rate = np.mean([r["collision"] for r in results]) if results else 0.0
    timeout_rate = np.mean([r["timeout"] for r in results]) if results else 0.0
    avg_len = np.mean([r["length"] for r in results]) if results else 0.0
    avg_ret = np.mean([r["return"] for r in results]) if results else 0.0
    avg_goal_dist = np.nanmean([r["final_goal_dist"] for r in results]) if results else np.nan
    avg_min_lidar = np.mean([r["min_lidar_mean"] for r in results]) if results else 0.0
    min_lidar_overall = np.min([r["min_lidar_min"] for r in results]) if results else 0.0
    avg_velocity = np.mean([r["velocity_mean"] for r in results]) if results else 0.0
    avg_recovery = np.mean([r["recovery_steps"] for r in results]) if results else 0.0
    avg_emergency = np.mean([r["emergency_steps"] for r in results]) if results else 0.0
    avg_near_goal = np.mean([r["near_goal_steps"] for r in results]) if results else 0.0

    std_len = np.std([r["length"] for r in results]) if results else 0.0
    std_ret = np.std([r["return"] for r in results]) if results else 0.0

    summary = {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "avg_length": avg_len,
        "std_length": std_len,
        "avg_return": avg_ret,
        "std_return": std_ret,
        "avg_final_goal_dist": avg_goal_dist,
        "avg_min_lidar": avg_min_lidar,
        "min_lidar_overall": min_lidar_overall,
        "avg_velocity": avg_velocity,
        "avg_recovery_steps": avg_recovery,
        "avg_emergency_steps": avg_emergency,
        "avg_near_goal_steps": avg_near_goal,
        "n_episodes": n_episodes,
        "eval_time": eval_time,
    }

    print(f"\n{'=' * 60}")
    print("DWA Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"success_rate      : {success_rate:.3f}")
    print(f"collision_rate    : {collision_rate:.3f}")
    print(f"timeout_rate      : {timeout_rate:.3f}")
    print(f"avg_length        : {avg_len:.1f} ± {std_len:.1f}")
    print(f"avg_return        : {avg_ret:.3f} ± {std_ret:.3f}")
    print(f"avg_final_dist    : {avg_goal_dist:.2f} m")
    print(f"avg_min_lidar     : {avg_min_lidar:.2f} m")
    print(f"min_lidar_overall : {min_lidar_overall:.2f} m")
    print(f"avg_velocity      : {avg_velocity:.3f}")
    print(f"avg_recovery_steps: {avg_recovery:.1f}")
    print(f"avg_emergency     : {avg_emergency:.1f}")
    print(f"avg_near_goal     : {avg_near_goal:.1f}")
    print(f"eval_time         : {eval_time:.1f} s")
    print(f"{'=' * 60}\n")

    return summary


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    env_cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\Builds_360\Project_1.exe",
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=367,
        lidar_dim=360,
        reach_goal_radius=0.6,   # 改为 0.6
        max_steps=350,
        progress_gain=2.5,
        time_penalty=-0.005,
        collision_penalty=-8.0,
        success_bonus=80.0,
        timeout_penalty=-15.0,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.15,
        action_l2_penalty=-0.0005,
    )

    policy_cfg = DWAConfig(
        lidar_dim=360,
        lidar_max_distance=10.0,
    )

    summary = evaluate_dwa(env_cfg, policy_cfg, n_episodes=50)
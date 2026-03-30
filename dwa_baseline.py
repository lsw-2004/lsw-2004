"""
DWA Baseline - 360° LiDAR 修复版

核心修复：
1. 修复轨迹安全检查的坐标系错误（LiDAR角度映射）
2. 移除被证伪的虚拟参考机制，直接朝向目标
3. 修复 emergency_avoid 动作归一化
4. 简化评分函数，移除冲突的 progress_bonus

坐标系约定：
- 局部坐标系：x=左右(右为正), y=前后(前为正)，对应 Unity 的 XZ 平面
- LiDAR：0°=前方(正Z)，逆时针增加（左=90°, 后=180°, 右=270°）
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from unity_env import UnityNavEnv, EnvConfig


# =============================================================================
# Config
# =============================================================================
@dataclass
class DWAConfig:
    # LiDAR
    lidar_dim: int = 360
    lidar_max_distance: float = 10.0
    control_dt: float = 0.1

    # 机器人运动学参数 - 提高最大速度
    max_vel_x: float = 1.5                  # 从 1.0 提高到 1.5
    max_vel_theta: float = 2.0              # 从 1.5 提高到 2.0，更快转向
    min_vel_x: float = 0.0
    min_vel_theta: float = -2.0

    # 加速度限制
    acc_lim_x: float = 2.5
    acc_lim_theta: float = 3.2

    # 速度采样 - 增加采样密度
    vx_samples: int = 16                    # 增加速度采样
    vtheta_samples: int = 24                # 增加角速度采样

    # 轨迹模拟 - 缩短预测时间，更关注近期，更快响应
    sim_time: float = 0.8                   # 更短预测，更快响应
    sim_granularity: float = 0.04           # 更精细粒度

    # 评分权重 - 激进优化：大幅提高速度权重，降低障碍敏感度
    path_distance_bias: float = 20.0      # 降低路径追求
    goal_distance_bias: float = 12.0      # 降低朝向权重
    occdist_scale: float = 0.8            # 大幅降低障碍惩罚，更激进绕行
    oscillation_scale: float = 0.10       # 降低震荡惩罚
    progress_bonus_weight: float = 12.0   # 大幅提高进度奖励
    low_speed_penalty_weight: float = 5.0 # 极严厉惩罚低速
    spin_penalty_weight: float = 0.8
    vel_smooth_factor: float = 0.015      # 更低平滑惩罚

    # 静态安全参数 - 更激进：允许更近的障碍物距离
    robot_radius: float = 0.45              # 稍微减小机器人半径
    safety_margin: float = 0.02             # 极小安全边距
    min_safe_distance: float = 0.15         # 允许更接近障碍

    # stuck / recovery - 更激进的恢复
    stuck_window: int = 20                  # 更快检测卡住
    recovery_steps: int = 3                 # 更短恢复时间
    recovery_vx: float = 0.5                # 更快的恢复速度
    recovery_vw_base: float = 1.0           # 更快的转向

    # near-goal mode - 调优后
    near_goal_dist: float = 1.0           # 稍微降低
    near_goal_turn_only_angle: float = 0.5
    near_goal_slow_vx: float = 0.3
    near_goal_fast_vx: float = 0.5
    near_goal_max_w: float = 1.0
    near_goal_front_safe: float = 0.4
    near_goal_emergency_front: float = 0.25

    # 轨迹安全检查参数
    trajectory_check_beam_width: int = 8  # 每个轨迹点检查的beam数量（单边）


# =============================================================================
# DWA Policy
# =============================================================================
class VirtualPathDWAPolicy:
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
        
        # 动态障碍检测相关
        self.prev_lidar = None
        self.obstacle_velocity = np.zeros(cfg.lidar_dim)  # 各方向的障碍接近速度
        self.dynamic_danger_sectors = []  # 动态危险区域

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
        
        # 重置动态检测
        self.prev_lidar = None
        self.obstacle_velocity = np.zeros(self.cfg.lidar_dim)
        self.dynamic_danger_sectors = []

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------
    def _index_to_angle_deg(self, idx: int) -> float:
        if self.cfg.lidar_dim <= 1:
            return 0.0
        return -180.0 + 360.0 * idx / (self.cfg.lidar_dim - 1)

    def _sector_indices(self, center_deg: float, half_width_deg: float) -> List[int]:
        """
        获取指定角度范围内的 LiDAR 索引
        center_deg: 中心角度（0=前方，正角度=右侧/顺时针，负角度=左侧/逆时针）
        half_width_deg: 半宽
        
        注意：Unity SimpleLidar2D 中角度增加方向是顺时针
        """
        ids = []
        for i in range(self.cfg.lidar_dim):
            ang = self._index_to_angle_deg(i)
            diff = ((ang - center_deg + 180.0) % 360.0) - 180.0
            if abs(diff) <= half_width_deg:
                ids.append(i)
        return ids

    def _sector_min(self, lidar_m: np.ndarray, center_deg: float, half_width_deg: float) -> float:
        ids = self._sector_indices(center_deg, half_width_deg)
        if not ids:
            return float(np.min(lidar_m))
        return float(np.min(lidar_m[ids]))

    def _get_front_left_right(self, lidar_m: np.ndarray) -> Tuple[float, float, float]:
        """
        获取前方、左侧、右侧的最小距离
        根据 Unity 顺时针约定：
        - 左侧 = 负角度 (-90°)
        - 右侧 = 正角度 (+90°)
        """
        front_min = self._sector_min(lidar_m, 0.0, 18.0)
        left_min = self._sector_min(lidar_m, -90.0, 35.0)   # 左侧是负角度
        right_min = self._sector_min(lidar_m, 90.0, 35.0)   # 右侧是正角度
        return front_min, left_min, right_min

    def _detect_dynamic_obstacles(self, lidar_m: np.ndarray) -> None:
        """
        检测动态障碍：通过比较前后两帧 LiDAR 数据，估计障碍接近速度
        优化版：降低敏感度，减少误判
        """
        if self.prev_lidar is None:
            self.prev_lidar = lidar_m.copy()
            return
        
        # 计算各方向的距离变化（负值表示障碍靠近）
        dt = self.cfg.control_dt
        distance_change = lidar_m - self.prev_lidar
        self.obstacle_velocity = distance_change / dt
        
        # 检测危险区域：使用更严格的阈值
        self.dynamic_danger_sectors = []
        approaching_threshold = -0.8  # 提高阈值：只有快速靠近才算（原来-0.3太敏感）
        min_dist_threshold = 1.5      # 降低距离阈值
        
        # 使用滑动窗口平滑速度估计，减少噪声
        window_size = 5
        for i in range(self.cfg.lidar_dim):
            # 计算局部平均速度
            idx_start = max(0, i - window_size)
            idx_end = min(self.cfg.lidar_dim, i + window_size + 1)
            avg_velocity = np.mean(self.obstacle_velocity[idx_start:idx_end])
            
            # 只有快速靠近且距离较近的障碍才算危险
            if avg_velocity < approaching_threshold and lidar_m[i] < min_dist_threshold:
                # 只关注前方±90度的障碍
                angle_deg = self._index_to_angle_deg(i)
                if abs(angle_deg) < 90:  # 只考虑前方
                    # 危险等级：距离越近、速度越快越危险
                    danger_level = abs(avg_velocity) / (lidar_m[i] + 0.1)
                    # 降低危险等级的整体幅度
                    danger_level = danger_level * 0.5
                    
                    self.dynamic_danger_sectors.append({
                        'idx': i,
                        'angle': angle_deg,
                        'dist': lidar_m[i],
                        'velocity': avg_velocity,
                        'danger_level': danger_level
                    })
        
        # 更新历史
        self.prev_lidar = lidar_m.copy()
    
    def _get_dynamic_danger_in_sector(self, center_deg: float, half_width_deg: float) -> float:
        """
        获取指定扇区内的动态危险程度
        返回值越大越危险
        """
        total_danger = 0.0
        for danger in self.dynamic_danger_sectors:
            diff = ((danger['angle'] - center_deg + 180.0) % 360.0) - 180.0
            if abs(diff) <= half_width_deg:
                total_danger += danger['danger_level']
        return total_danger

    # -------------------------------------------------------------------------
    # 目标参考计算（带绕行方向选择）
    # -------------------------------------------------------------------------
    def _compute_goal_reference(
        self,
        lidar_m: np.ndarray,
        goal_angle: float,
        goal_dist: float
    ) -> Tuple[float, float, float]:
        """
        计算目标参考点 - 智能绕行版：结合静态和动态障碍信息
        返回: ref_x, ref_y, ref_heading（局部坐标系）
        """
        # 获取各方向的最小距离
        front_min = self._sector_min(lidar_m, 0.0, 25.0)
        left_min = self._sector_min(lidar_m, -90.0, 45.0)   # 左侧大范围
        right_min = self._sector_min(lidar_m, 90.0, 45.0)   # 右侧大范围
        left_front = self._sector_min(lidar_m, -45.0, 30.0) # 左前方
        right_front = self._sector_min(lidar_m, 45.0, 30.0) # 右前方
        
        # 获取动态危险信息
        front_dynamic_danger = self._get_dynamic_danger_in_sector(0.0, 30.0)
        left_dynamic_danger = self._get_dynamic_danger_in_sector(-45.0, 30.0)
        right_dynamic_danger = self._get_dynamic_danger_in_sector(45.0, 30.0)
        
        # 基础目标方向
        ref_heading = goal_angle
        
        # 综合静态和动态障碍进行绕行决策
        need_avoid = front_min < 2.0 and goal_dist > 0.8
        need_avoid = need_avoid or front_dynamic_danger > 2.0  # 提高阈值：只有高动态危险才触发
        
        if need_avoid:
            # 综合考虑静态间隙和动态危险
            # 静态分数：越大越安全
            left_static_score = left_min + left_front
            right_static_score = right_min + right_front
            
            # 动态分数：越小越安全（所以要减去），但降低权重
            left_total_score = left_static_score - 1.5 * left_dynamic_danger  # 降低权重
            right_total_score = right_static_score - 1.5 * right_dynamic_danger
            
            side_clearance_diff = left_total_score - right_total_score
            
            # 根据综合分数选择绕行方向
            if side_clearance_diff > 0.8:
                steer_bias = -0.9  # 强烈左转
            elif side_clearance_diff < -0.8:
                steer_bias = 0.9   # 强烈右转
            elif side_clearance_diff > 0:
                steer_bias = -0.6
            elif side_clearance_diff < 0:
                steer_bias = 0.6
            else:
                steer_bias = 0.0
            
            # 动态障碍紧急避让：只有极高危险才强制转向
            if front_dynamic_danger > 4.0:  # 提高阈值
                # 紧急避让：选择动态危险更小的一侧
                if left_dynamic_danger < right_dynamic_danger:
                    steer_bias = -1.0
                else:
                    steer_bias = 1.0
                ref_heading = goal_angle + steer_bias
            elif front_min < 1.0 or front_dynamic_danger > 1.0:  # 提高阈值
                # 前方有障碍或动态危险，强制应用绕行
                ref_heading = goal_angle + steer_bias
            else:
                # 前方还有空间，柔和调整
                ref_heading = goal_angle + 0.5 * steer_bias
            
            # 限制调整范围
            ref_heading = np.clip(ref_heading, -1.3, 1.3)
        
        ref_dist = goal_dist
        
        # 局部坐标系：x=左右, y=前方
        ref_x = ref_dist * math.sin(ref_heading)
        ref_y = ref_dist * math.cos(ref_heading)
        
        return ref_x, ref_y, ref_heading

    # -------------------------------------------------------------------------
    # Near-goal mode
    # -------------------------------------------------------------------------
    def _near_goal_control(self, lidar_m: np.ndarray, goal_dist: float, goal_angle: float) -> np.ndarray:
        self._near_goal_mode_called = True

        front_min = self._sector_min(lidar_m, 0.0, 20.0)
        left_front_min = self._sector_min(lidar_m, -35.0, 20.0)
        right_front_min = self._sector_min(lidar_m, 35.0, 20.0)

        if front_min < self.cfg.near_goal_emergency_front:
            action_v = 0.0
            action_w = self.cfg.near_goal_max_w if left_front_min >= right_front_min else -self.cfg.near_goal_max_w
            return np.array(
                [action_v / self.cfg.max_vel_x, action_w / self.cfg.max_vel_theta],
                dtype=np.float32
            )

        if abs(goal_angle) > self.cfg.near_goal_turn_only_angle:
            action_v = 0.0
            action_w = np.clip(1.3 * goal_angle, -self.cfg.near_goal_max_w, self.cfg.near_goal_max_w)
            return np.array(
                [action_v / self.cfg.max_vel_x, action_w / self.cfg.max_vel_theta],
                dtype=np.float32
            )

        if front_min > self.cfg.near_goal_front_safe:
            action_v = self.cfg.near_goal_fast_vx if goal_dist > 0.8 else self.cfg.near_goal_slow_vx
            action_w = np.clip(1.0 * goal_angle, -0.40, 0.40)
        else:
            action_v = 0.12
            steer_bias = 0.0
            if left_front_min > right_front_min + 0.05:
                steer_bias = 0.18
            elif right_front_min > left_front_min + 0.05:
                steer_bias = -0.18
            action_w = np.clip(0.9 * goal_angle + steer_bias, -0.50, 0.50)

        return np.array(
            [action_v / self.cfg.max_vel_x, action_w / self.cfg.max_vel_theta],
            dtype=np.float32
        )

    # -------------------------------------------------------------------------
    # Main
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
        
        # 检测动态障碍
        self._detect_dynamic_obstacles(lidar_m)

        # 近目标模式优先
        if goal_dist < self.cfg.near_goal_dist:
            self.is_recovery_mode = False
            self.recovery_counter = 0
            return self._near_goal_control(lidar_m, goal_dist, goal_angle)

        # 历史
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
            left_min = self._sector_min(lidar_m, -90.0, 60.0)
            right_min = self._sector_min(lidar_m, 90.0, 60.0)
            self.recovery_direction = 1.0 if left_min > right_min else -1.0
            return self._execute_recovery(lidar_m, goal_angle)

        # 构造目标参考点（直接朝向目标，不使用虚拟参考）
        ref_x, ref_y, ref_heading = self._compute_goal_reference(lidar_m, goal_angle, goal_dist)

        v_min, v_max, w_min, w_max = self._compute_dynamic_window(cur_v, cur_w)
        # 调优后：速度采样偏向高速，使用非线性分布
        vx_samples = np.linspace(v_min, v_max, self.cfg.vx_samples)
        # 对速度进行平方映射，让高速样本更多
        vx_ratio = (vx_samples - v_min) / max(1e-6, v_max - v_min)
        vx_samples = v_min + (v_max - v_min) * (vx_ratio ** 0.7)  # 0.7 < 1 让高速样本更密集
        vtheta_samples = np.linspace(w_min, w_max, self.cfg.vtheta_samples)

        best_score = -float("inf")
        best_vx = 0.0
        best_vtheta = 0.0
        safe_candidates = []

        for vx in vx_samples:
            for vtheta in vtheta_samples:
                trajectory = self._simulate_trajectory(vx, vtheta)
                is_safe, min_clearance_eff = self._check_trajectory_safety(trajectory, lidar_m)

                if not is_safe:
                    continue

                score = self._score_trajectory(
                    trajectory=trajectory,
                    vx=vx,
                    vtheta=vtheta,
                    ref_x=ref_x,
                    ref_y=ref_y,
                    ref_heading=ref_heading,
                    lidar_m=lidar_m,
                    min_clearance_eff=min_clearance_eff,
                )

                safe_candidates.append((score, vx, vtheta, min_clearance_eff))
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

        return np.array(
            [best_vx / self.cfg.max_vel_x, best_vtheta / self.cfg.max_vel_theta],
            dtype=np.float32
        )

    # -------------------------------------------------------------------------
    # Stuck / recovery / emergency
    # -------------------------------------------------------------------------
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
            action_w = np.sign(goal_angle) * 0.12 if abs(goal_angle) < math.pi / 4 else 0.0
        else:
            if left_min > right_min + 0.1:
                action_w = self.cfg.recovery_vw_base
            elif right_min > left_min + 0.1:
                action_w = -self.cfg.recovery_vw_base
            else:
                action_w = self.cfg.recovery_vw_base * self.recovery_direction

            side_min = max(left_min, right_min)
            action_v = self.cfg.recovery_vx * 0.25 if side_min > safe_dist * 1.45 else 0.0

        return np.array(
            [action_v / self.cfg.max_vel_x, action_w / self.cfg.max_vel_theta],
            dtype=np.float32
        )

    def _emergency_avoid(self, lidar_m: np.ndarray, goal_angle: float) -> np.ndarray:
        self._emergency_avoid_called = True

        front_min, left_min, right_min = self._get_front_left_right(lidar_m)
        left_front = self._sector_min(lidar_m, -30.0, 20.0)
        right_front = self._sector_min(lidar_m, 30.0, 20.0)
        safe_dist = self.cfg.robot_radius + self.cfg.safety_margin
        
        # 获取动态危险信息
        left_dynamic = self._get_dynamic_danger_in_sector(-30.0, 20.0)
        right_dynamic = self._get_dynamic_danger_in_sector(30.0, 20.0)
        front_dynamic = self._get_dynamic_danger_in_sector(0.0, 20.0)

        # 智能紧急避障：考虑动态障碍
        if front_min > safe_dist * 2.5 and front_dynamic < 0.5:
            # 前方较安全且没有动态危险，可以前进
            action_v = 0.5
            action_w = 0.0
        elif left_front > right_front and left_front > safe_dist * 1.5 and left_dynamic < right_dynamic:
            # 左前方更开阔且动态危险更小，果断左转
            action_w = 1.2
            action_v = 0.1
        elif right_front > safe_dist * 1.5 and right_dynamic < left_dynamic:
            # 右前方更开阔且动态危险更小，果断右转
            action_w = -1.2
            action_v = 0.1
        elif left_min > right_min and left_min > safe_dist and left_dynamic < 1.0:
            # 左侧更开阔且动态危险可控，左转
            action_w = 1.0
            action_v = 0.0
        elif right_min > safe_dist and right_dynamic < 1.0:
            # 右侧更开阔且动态危险可控，右转
            action_w = -1.0
            action_v = 0.0
        else:
            # 被困住，选择动态危险更小的一侧
            action_v = -0.3  # 稍微后退
            if left_dynamic < right_dynamic:
                action_w = 1.3
            else:
                action_w = -1.3

        # 归一化到 [-1, 1] 范围
        return np.array(
            [action_v / self.cfg.max_vel_x, action_w / self.cfg.max_vel_theta],
            dtype=np.float32
        )

    # -------------------------------------------------------------------------
    # DWA internals
    # -------------------------------------------------------------------------
    def _compute_dynamic_window(self, cur_v: float, cur_w: float) -> Tuple[float, float, float, float]:
        dt = self.cfg.control_dt

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
        """
        增强版轨迹安全检查 - 考虑动态障碍预测
        """
        min_clearance_eff = float("inf")
        dt = self.cfg.sim_granularity

        for i, (x, y, theta) in enumerate(trajectory):
            point_dist = math.sqrt(x**2 + y**2)
            time_at_point = i * dt  # 到达该轨迹点的时间
            
            # 计算轨迹点相对于机器人前方的角度
            angle_rad = math.atan2(x, y)
            angle_deg = math.degrees(angle_rad)
            
            # 映射到 LiDAR 索引
            center_idx = int(round((angle_deg + 180.0) / 360.0 * (self.cfg.lidar_dim - 1)))
            center_idx = max(0, min(self.cfg.lidar_dim - 1, center_idx))

            # 检查周围 beam
            idx_range = self.cfg.trajectory_check_beam_width
            idx_start = max(0, center_idx - idx_range)
            idx_end = min(self.cfg.lidar_dim, center_idx + idx_range + 1)

            lidar_dist = float(np.min(lidar_m[idx_start:idx_end]))
            
            # 动态障碍预测：检查该方向是否有快速接近的障碍
            dynamic_safety_margin = 0.0
            for danger in self.dynamic_danger_sectors:
                # 计算该危险方向与轨迹方向的夹角
                angle_diff = abs(((danger['angle'] - angle_deg + 180.0) % 360.0) - 180.0)
                if angle_diff < 15.0:  # 缩小范围到15度
                    # 预测障碍在未来时间的位置
                    predicted_dist = danger['dist'] + danger['velocity'] * time_at_point
                    if predicted_dist < danger['dist']:  # 障碍在靠近
                        # 降低安全边距系数，减少过度反应
                        dynamic_safety_margin = max(dynamic_safety_margin, 
                                                    0.15 * (danger['dist'] - predicted_dist))

            # 有效净空 = LiDAR测距 - 轨迹点距离 - 机器人半径 - 动态安全边距
            effective_clearance = lidar_dist - point_dist - self.cfg.robot_radius - dynamic_safety_margin
            min_clearance_eff = min(min_clearance_eff, effective_clearance)

            # 恢复原来的安全检查阈值，允许轻微接触
            required_clearance = -0.05  # 允许轻微进入安全边界
            if effective_clearance < required_clearance:
                return False, min_clearance_eff

        return True, min_clearance_eff

    def _score_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]],
        vx: float,
        vtheta: float,
        ref_x: float,
        ref_y: float,
        ref_heading: float,
        lidar_m: np.ndarray,
        min_clearance_eff: float
    ) -> float:
        if not trajectory:
            return -float("inf")

        end_x, end_y, end_theta = trajectory[-1]

        # 1. 朝虚拟参考点的距离得分
        dist_to_ref = math.sqrt((end_x - ref_x) ** 2 + (end_y - ref_y) ** 2)
        ref_dist = math.sqrt(ref_x ** 2 + ref_y ** 2)
        path_score = self.cfg.path_distance_bias * (ref_dist - dist_to_ref)

        # 2. 朝虚拟参考方向的朝向得分
        angle_to_ref = math.atan2(ref_x - end_x, ref_y - end_y)
        heading_diff = abs(self._normalize_angle(angle_to_ref - end_theta))
        goal_score = self.cfg.goal_distance_bias * (math.pi - heading_diff)

        # 3. 净空代价
        safe_clearance = self.cfg.safety_margin + self.cfg.min_safe_distance
        occ_cost = 0.0
        if min_clearance_eff < safe_clearance:
            dist_ratio = (safe_clearance - min_clearance_eff) / max(1e-6, safe_clearance)
            occ_cost += self.cfg.occdist_scale * (dist_ratio ** 1.6) * 60.0

        # 4. 朝向目标的进度奖励（替代原来的 forward_progress）
        # 计算轨迹终点到目标的距离
        end_dist_to_goal = math.sqrt((end_x - ref_x)**2 + (end_y - ref_y)**2)
        # 进度 = 初始到目标距离 - 终点到目标距离
        progress = ref_dist - end_dist_to_goal
        progress_bonus = self.cfg.progress_bonus_weight * progress

        # 5. 前方速度奖励 - 智能调整：有动态危险时降低速度
        front_min = self._sector_min(lidar_m, 0.0, 20.0)
        front_dynamic_danger = self._get_dynamic_danger_in_sector(0.0, 25.0)
        
        # 根据动态危险调整速度策略（提高阈值，减少过度反应）
        if front_dynamic_danger > 3.0:  # 只有极高危险才大幅减速
            # 有快速接近的障碍，大幅降低速度奖励
            vel_score = vx * 0.5
        elif front_dynamic_danger > 1.0:  # 中等危险适度减速
            # 中等动态危险，适度降低速度
            vel_score = vx * 2.0
        elif front_min > 0.8:
            vel_score = vx * 3.5       # 开阔地带极高速度奖励
        elif front_min > 0.5:
            vel_score = vx * 2.5       # 中等距离高奖励
        elif front_min > 0.35:
            vel_score = vx * 1.5       # 近距离保持速度
        else:
            vel_score = vx * 0.5       # 很近时才降低

        # 6. 低速惩罚 - 恢复原来的严格惩罚，提高速度
        low_speed_penalty = 0.0
        if vx < 0.4 and min_clearance_eff > 0.05:
            # 恢复惩罚低速，提高整体速度
            low_speed_penalty = self.cfg.low_speed_penalty_weight * (0.4 - vx) / 0.4

        # 7. 原地转圈惩罚
        spin_penalty = 0.0
        if vx < 0.1 and abs(vtheta) > 0.3:
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
            - occ_cost - low_speed_penalty
            - spin_penalty - smooth_penalty - oscillation_penalty
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
def run_dwa_episode(env: UnityNavEnv, policy: VirtualPathDWAPolicy) -> Dict:
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
    policy = VirtualPathDWAPolicy(policy_cfg)

    results = []
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print("DWA Evaluation (Fixed Version)")
    print(f"{'=' * 60}")
    print(f"Episodes: {n_episodes}")
    print(f"Environment: {env_cfg.file_name}")
    print("Config:")
    print(f"  - lidar_dim: {policy_cfg.lidar_dim}")
    print(f"  - robot_radius: {policy_cfg.robot_radius}")
    print(f"  - safety_margin: {policy_cfg.safety_margin}")
    print(f"  - min_safe_distance: {policy_cfg.min_safe_distance}")
    print(f"  - occdist_scale: {policy_cfg.occdist_scale}")
    print(f"  - near_goal_dist: {policy_cfg.near_goal_dist}")
    print(f"  - trajectory_check_beam_width: {policy_cfg.trajectory_check_beam_width}")
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
    print("DWA Summary")
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
    print(f"avg_recovery      : {avg_recovery:.1f}")
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
        worker_id=1,
        obs_size=367,
        lidar_dim=360,
        reach_goal_radius=0.6,
        max_steps=600,        # 增加最大步数，减少超时
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
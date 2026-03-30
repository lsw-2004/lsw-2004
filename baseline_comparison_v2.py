"""
基线算法对比评估模块 V2 - 基于论文标准的成熟实现

包含:
1. DWA (Dynamic Window Approach) - Fox et al., 1997 标准实现 + 推进优先改进
2. ORCA (Optimal Reciprocal Collision Avoidance) - van den Berg et al., 2011
3. APF (Artificial Potential Field) - Khatib, 1986 标准实现 + 逃逸机制
4. VFH+ (Vector Field Histogram+) - Ulrich & Borenstein, 1998
5. Time Elastic Band (TEB) 简化版 - 基于轨迹优化

参考论文:
- Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach to collision avoidance.
- van den Berg, J., et al. (2011). Reciprocal n-Body Collision Avoidance. ICRA.
- Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots.
- Ulrich, I., & Borenstein, J. (1998). VFH+: Reliable obstacle avoidance for fast mobile robots.
- Rösmann, C., et al. (2017). Efficient trajectory optimization using a sparse model.

用于与 PPO 强化学习方法进行对比实验。
"""
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from unity_env import UnityNavEnv, EnvConfig


# =============================================================================
# DWA (Dynamic Window Approach) - Fox et al., 1997 标准实现
# =============================================================================
@dataclass
class DWAConfig:
    """DWA 算法配置 - 保守避障 + 适度推进
    
    核心原则:
    1. 避障优先 - 先保证不碰撞，再谈推进
    2. 保守膨胀 - 给动态障碍物留安全余量
    3. 适度推进 - 在安全前提下保持前进
    """
    
    # 机器人运动学参数
    max_vel_x: float = 1.0          # 最大前进速度 (m/s)
    max_vel_theta: float = 1.5      # 最大角速度 (rad/s)
    min_vel_x: float = 0.0          # 允许停止，避免强制推进导致碰撞
    min_vel_theta: float = -1.5     # 最小角速度
    
    # 加速度限制
    acc_lim_x: float = 2.5          # 最大线加速度 (m/s^2)
    acc_lim_theta: float = 3.2      # 最大角加速度 (rad/s^2)
    
    # 速度采样 - 适中的采样密度
    vx_samples: int = 10            # 线速度采样数
    vtheta_samples: int = 20        # 角速度采样数
    
    # 轨迹模拟参数
    sim_time: float = 1.5           # 轨迹模拟时间 (s)
    sim_granularity: float = 0.05   # 模拟步长 (s)
    
    # 评分权重
    path_distance_bias: float = 32.0    # 目标距离权重
    goal_distance_bias: float = 20.0    # 目标方向权重
    occdist_scale: float = 2.5          # 障碍物代价权重 - 提高以增强避障
    oscillation_scale: float = 0.3      # 震荡惩罚
    
    # 推进权重 - 适度
    progress_bonus_weight: float = 3.0      # 前向位移奖励
    low_speed_penalty_weight: float = 0.5   # 低速惩罚 - 降低
    spin_penalty_weight: float = 1.0        # 原地转圈惩罚
    
    # 障碍物参数 - 保守配置
    robot_radius: float = 0.25      # 机器人半径 (m)
    inflation_radius: float = 1.0   # 膨胀半径 (m) - 增大安全距离
    safety_margin: float = 0.15     # 额外安全边距
    min_safe_distance: float = 0.5  # 最小安全距离
    
    # 前向预测
    forward_point_distance: float = 0.325  # 前向点距离
    
    # 速度平滑
    vel_smooth_factor: float = 0.1  # 平滑惩罚
    
    # Stuck Recovery 配置
    stuck_velocity_threshold: float = 0.08   # 速度低于此值计入stuck
    stuck_progress_threshold: float = 0.03   # 目标距离下降率阈值
    stuck_window: int = 15                   # stuck检测窗口大小
    recovery_steps: int = 15                 # recovery持续步数
    recovery_vx: float = 0.20                # recovery模式前进速度 - 降低
    recovery_vw_base: float = 0.8            # recovery模式基础角速度


class DWAPolicyV2:
    """
    DWA 策略实现 - 保守避障版本
    
    核心原则:
    1. 安全第一 - 任何可能导致碰撞的轨迹直接排除
    2. 保守膨胀 - 考虑动态障碍物的不确定性
    3. 适度推进 - 在安全前提下保持前进
    
    改进:
    - 更严格的碰撞检测
    - 考虑机器人实际尺寸的安全距离
    - 改进的 stuck recovery
    """
    
    def __init__(self, cfg: DWAConfig):
        self.cfg = cfg
        self.prev_vx = 0.0
        self.prev_vtheta = 0.0
        self.prev_actions = []
        
        # Stuck detection & recovery
        self.velocity_history = []
        self.goal_dist_history = []
        self.is_recovery_mode = False
        self.recovery_counter = 0
        self.recovery_direction = 0.0
        self.total_steps = 0
        self.consecutive_low_vel = 0  # 连续低速计数
        
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
        
    def act(self, obs: np.ndarray) -> np.ndarray:
        """根据观测选择动作 - 保守避障版 DWA"""
        self.total_steps += 1
        
        # 解析观测
        lidar_norm = obs[:180]
        goal_dist = float(obs[182]) * 30.0
        goal_angle = float(obs[183]) * math.pi
        cur_v = float(obs[184]) * self.cfg.max_vel_x
        cur_w = float(obs[185]) * self.cfg.max_vel_theta
        
        # LiDAR 转换为米
        lidar_m = lidar_norm * 10.0
        
        # 更新历史
        self.velocity_history.append(cur_v)
        self.goal_dist_history.append(goal_dist)
        if len(self.velocity_history) > self.cfg.stuck_window:
            self.velocity_history.pop(0)
            self.goal_dist_history.pop(0)
        
        # 检测是否卡住
        is_stuck = self._detect_stuck()
        
        # Recovery 模式
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
        
        # 如果检测到卡住，进入恢复模式
        if is_stuck and not self.is_recovery_mode:
            self.is_recovery_mode = True
            self.recovery_counter = 0
            left_min = float(np.min(lidar_m[0:60]))
            right_min = float(np.min(lidar_m[120:180]))
            self.recovery_direction = 1.0 if left_min > right_min else -1.0
            return self._execute_recovery(lidar_m, goal_angle)
        
        # === 正常 DWA 规划 ===
        # 计算动态窗口
        v_min, v_max, w_min, w_max = self._compute_dynamic_window(cur_v, cur_w)
        
        # 生成候选速度
        vx_samples = np.linspace(v_min, v_max, self.cfg.vx_samples)
        vtheta_samples = np.linspace(w_min, w_max, self.cfg.vtheta_samples)
        
        # 寻找最优速度
        best_score = -float('inf')
        best_vx = 0.0  # 默认停止
        best_vtheta = 0.0
        best_is_safe = False
        
        # 收集所有安全的候选
        safe_candidates = []
        
        for vx in vx_samples:
            for vtheta in vtheta_samples:
                # 模拟轨迹
                trajectory = self._simulate_trajectory(vx, vtheta)
                
                # 先检查是否安全
                is_safe, min_dist = self._check_trajectory_safety(trajectory, lidar_m)
                
                if not is_safe:
                    continue  # 直接排除不安全的轨迹
                
                # 计算得分
                score = self._score_trajectory(
                    trajectory, vx, vtheta, 
                    goal_dist, goal_angle, lidar_m, min_dist
                )
                
                safe_candidates.append((score, vx, vtheta, min_dist))
                
                if score > best_score:
                    best_score = score
                    best_vx = vx
                    best_vtheta = vtheta
                    best_is_safe = True
        
        # 如果没有安全的候选，执行紧急避障
        if not safe_candidates:
            return self._emergency_avoid(lidar_m, goal_angle)
        
        # 保存状态
        self.prev_vx = best_vx
        self.prev_vtheta = best_vtheta
        
        # 更新连续低速计数
        if best_vx < 0.05:
            self.consecutive_low_vel += 1
        else:
            self.consecutive_low_vel = 0
        
        # 记录历史
        self.prev_actions.append((best_vx, best_vtheta))
        if len(self.prev_actions) > 10:
            self.prev_actions.pop(0)
        
        # 转换为归一化动作
        action_v = best_vx / self.cfg.max_vel_x
        action_w = best_vtheta / self.cfg.max_vel_theta
        
        return np.array([action_v, action_w], dtype=np.float32)
    
    def _detect_stuck(self) -> bool:
        """检测是否陷入停滞状态"""
        if len(self.velocity_history) < self.cfg.stuck_window:
            return False
        
        # 检测1：连续低速
        avg_vel = np.mean(self.velocity_history)
        if avg_vel < self.cfg.stuck_velocity_threshold:
            return True
        
        # 检测2：目标距离几乎没下降
        if len(self.goal_dist_history) >= self.cfg.stuck_window:
            progress = self.goal_dist_history[0] - self.goal_dist_history[-1]
            progress_rate = progress / max(self.goal_dist_history[0], 0.1)
            if progress_rate < self.cfg.stuck_progress_threshold:
                return True
        
        # 检测3：连续低速步数过多
        if self.consecutive_low_vel > 20:
            return True
        
        return False
    
    def _execute_recovery(self, lidar_m: np.ndarray, goal_angle: float) -> np.ndarray:
        """执行恢复动作：原地转向到安全方向"""
        # 检查前方安全距离
        front_min = float(np.min(lidar_m[75:105]))
        
        # 如果前方空间充足，缓慢前进
        if front_min > 1.2:
            action_v = self.cfg.recovery_vx * 0.5
            action_w = 0.0
        else:
            # 原地转向，找到最宽的方向
            left_min = float(np.min(lidar_m[30:90]))
            right_min = float(np.min(lidar_m[90:150]))
            
            # 优先转向目标方向，除非那边更窄
            if abs(goal_angle) < math.pi / 2:
                preferred_dir = np.sign(goal_angle)
                # 检查目标方向是否足够宽
                if goal_angle > 0 and left_min > 0.8:
                    action_w = self.cfg.recovery_vw_base
                elif goal_angle < 0 and right_min > 0.8:
                    action_w = -self.cfg.recovery_vw_base
                else:
                    # 目标方向不够宽，转向更宽的一侧
                    action_w = self.cfg.recovery_vw_base if left_min > right_min else -self.cfg.recovery_vw_base
            else:
                # 目标在侧面，转向更宽的一侧
                action_w = self.cfg.recovery_vw_base if left_min > right_min else -self.cfg.recovery_vw_base
            
            action_v = 0.0  # 原地转向
        
        return np.array([
            action_v / self.cfg.max_vel_x,
            action_w / self.cfg.max_vel_theta
        ], dtype=np.float32)
    
    def _emergency_avoid(self, lidar_m: np.ndarray, goal_angle: float) -> np.ndarray:
        """紧急避障：当没有安全轨迹时执行"""
        # 找到最宽的方向
        sector_width = 30  # 30度扇区
        best_heading = 0
        max_clearance = 0
        
        for i in range(0, 180, 10):
            left = max(0, i - sector_width // 2)
            right = min(180, i + sector_width // 2)
            clearance = float(np.min(lidar_m[left:right]))
            if clearance > max_clearance:
                max_clearance = clearance
                best_heading = (i - 90) * math.pi / 180  # 转换为弧度
        
        # 计算转向方向
        action_w = np.clip(best_heading / 0.5, -1.0, 1.0) * self.cfg.max_vel_theta
        
        # 只有前方足够宽才前进
        front_min = float(np.min(lidar_m[75:105]))
        if front_min > 1.0:
            action_v = min(0.3, front_min * 0.3)
        else:
            action_v = 0.0
        
        return np.array([
            action_v / self.cfg.max_vel_x,
            action_w / self.cfg.max_vel_theta
        ], dtype=np.float32)
    
    def _compute_dynamic_window(self, cur_v: float, cur_w: float) -> Tuple[float, float, float, float]:
        """计算动态窗口（考虑加速度限制）"""
        dt = 0.1  # 控制周期
        
        # 基于加速度限制的速度范围
        v_min_accel = cur_v - self.cfg.acc_lim_x * dt
        v_max_accel = cur_v + self.cfg.acc_lim_x * dt
        w_min_accel = cur_w - self.cfg.acc_lim_theta * dt
        w_max_accel = cur_w + self.cfg.acc_lim_theta * dt
        
        # 与运动学限制取交集
        v_min = max(self.cfg.min_vel_x, v_min_accel)
        v_max = min(self.cfg.max_vel_x, v_max_accel)
        w_min = max(self.cfg.min_vel_theta, w_min_accel)
        w_max = min(self.cfg.max_vel_theta, w_max_accel)
        
        return v_min, v_max, w_min, w_max
    
    def _simulate_trajectory(self, vx: float, vtheta: float) -> List[Tuple[float, float, float]]:
        """模拟给定速度下的轨迹"""
        trajectory = []
        x, y, theta = 0.0, 0.0, 0.0
        
        n_steps = int(self.cfg.sim_time / self.cfg.sim_granularity)
        for _ in range(n_steps):
            x += vx * self.cfg.sim_granularity * math.cos(theta)
            y += vx * self.cfg.sim_granularity * math.sin(theta)
            theta += vtheta * self.cfg.sim_granularity
            trajectory.append((x, y, theta))
        
        return trajectory
    
    def _check_trajectory_safety(self, trajectory: List[Tuple[float, float, float]], 
                                  lidar_m: np.ndarray) -> Tuple[bool, float]:
        """检查轨迹是否安全，返回 (是否安全, 最小距离)"""
        min_dist = float('inf')
        
        for x, y, theta in trajectory:
            # 计算点在 LiDAR 数据中的对应角度
            point_angle = math.atan2(x, y + 1e-6)
            point_dist = math.sqrt(x**2 + y**2)
            
            # LiDAR 索引: 0=左侧(-90°), 90=前方(0°), 179=右侧(+89°)
            lidar_idx = int(90 - math.degrees(point_angle))
            lidar_idx = max(0, min(179, lidar_idx))
            
            # 获取该方向周围的 LiDAR 距离 (取更大范围的最小值)
            idx_range = 5  # 扩大检测范围
            idx_start = max(0, lidar_idx - idx_range)
            idx_end = min(180, lidar_idx + idx_range + 1)
            lidar_dist = float(np.min(lidar_m[idx_start:idx_end]))
            
            # 计算到障碍物的距离
            dist_to_obs = lidar_dist - point_dist
            min_dist = min(min_dist, dist_to_obs)
            
            # 安全判定：必须留有足够的安全距离
            required_dist = self.cfg.robot_radius + self.cfg.safety_margin
            if dist_to_obs < required_dist:
                return False, min_dist
        
        return True, min_dist
    
    def _score_trajectory(self, trajectory: List[Tuple[float, float, float]], 
                          vx: float, vtheta: float,
                          goal_dist: float, goal_angle: float,
                          lidar_m: np.ndarray,
                          min_dist: float) -> float:
        """计算轨迹得分 - 安全优先版"""
        if not trajectory:
            return -float('inf')
        
        end_x, end_y, end_theta = trajectory[-1]
        
        # 1. 目标距离得分
        goal_x = goal_dist * math.sin(goal_angle)
        goal_y = goal_dist * math.cos(goal_angle)
        dist_to_goal = math.sqrt((end_x - goal_x)**2 + (end_y - goal_y)**2)
        path_score = self.cfg.path_distance_bias * (goal_dist - dist_to_goal)
        
        # 2. 目标方向得分
        angle_to_goal = math.atan2(goal_y - end_y, goal_x - end_x)
        heading_diff = abs(self._normalize_angle(angle_to_goal - end_theta))
        goal_score = self.cfg.goal_distance_bias * (math.pi - heading_diff)
        
        # 3. 障碍物代价 (基于最小距离)
        # 距离越近，代价越高
        safe_dist = self.cfg.robot_radius + self.cfg.safety_margin + self.cfg.min_safe_distance
        if min_dist < safe_dist:
            occ_cost = self.cfg.occdist_scale * ((safe_dist - min_dist) / safe_dist) ** 2 * 100
        else:
            occ_cost = 0.0
        
        # 4. 前进奖励 (适度)
        forward_progress = end_y
        progress_bonus = self.cfg.progress_bonus_weight * max(0, forward_progress)
        
        # 5. 速度奖励 (适度)
        vel_score = vx * 0.5
        
        # 6. 低速惩罚 (轻微)
        low_speed_penalty = 0.0
        if vx < 0.1 and min_dist > safe_dist:
            low_speed_penalty = self.cfg.low_speed_penalty_weight * (0.1 - vx) / 0.1
        
        # 7. 原地转圈惩罚
        spin_penalty = 0.0
        if vx < 0.1 and abs(vtheta) > 0.5:
            spin_penalty = self.cfg.spin_penalty_weight * (abs(vtheta) / self.cfg.max_vel_theta)
        
        # 8. 平滑惩罚
        smooth_penalty = self.cfg.vel_smooth_factor * (
            abs(vx - self.prev_vx) + 0.5 * abs(vtheta - self.prev_vtheta)
        )
        
        # 9. 震荡惩罚
        oscillation_penalty = 0.0
        if len(self.prev_actions) >= 4:
            recent = self.prev_actions[-4:]
            if (recent[0][1] * recent[2][1] < 0 and recent[1][1] * recent[3][1] < 0):
                oscillation_penalty = self.cfg.oscillation_scale * abs(vtheta)
        
        # 总得分
        total_score = (path_score + goal_score + vel_score + progress_bonus
                       - occ_cost - low_speed_penalty - spin_penalty
                       - smooth_penalty - oscillation_penalty)
        
        return total_score
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# =============================================================================
# ORCA (Optimal Reciprocal Collision Avoidance) - van den Berg et al., 2011
# =============================================================================
@dataclass
class ORCAConfig:
    """ORCA 配置 - 针对动态走廊场景调优
    
    ORCA 核心思想:
    1. 为每个邻居计算速度障碍 (Velocity Obstacle)
    2. 在速度空间中选择最优速度，避免碰撞
    3. 保证多智能体避障的完备性
    
    针对你的场景调优:
    - 降低安全边距，允许更近距离通过
    - 增大目标导向权重，更敢于推进
    - 添加 stuck recovery 机制
    """
    
    # 机器人参数
    robot_radius: float = 0.25
    pedestrian_radius: float = 0.35
    safety_margin: float = 0.15       # 减小安全边距，允许更近距离通过
    
    # 速度限制
    max_vel_x: float = 1.0
    max_vel_theta: float = 1.5
    
    # ORCA 参数
    time_horizon: float = 2.0         # 预测时域
    neighbor_dist: float = 8.0        # 邻居检测距离
    max_neighbors: int = 5            # 最大邻居数
    
    # 采样参数
    n_speed_samples: int = 8          # 速度采样数
    n_heading_samples: int = 16       # 朝向采样数
    
    # 评分权重
    goal_weight: float = 2.5          # 目标导向权重
    speed_weight: float = 1.0         # 速度奖励
    heading_weight: float = 0.8       # 朝向奖励
    smooth_weight: float = 0.05       # 平滑惩罚
    
    # Stuck Recovery
    stuck_velocity_threshold: float = 0.12
    stuck_window: int = 10
    recovery_steps: int = 8
    recovery_vx: float = 0.35
    recovery_vw: float = 0.6
    
    # 静态障碍物 LiDAR 过滤
    lidar_max_range: float = 10.0
    static_stop_dist: float = 0.45    # 比原来更近才停止
    static_slow_dist: float = 1.0     # 比原来更近才减速


class ORCAPolicy:
    """
    ORCA 策略 - 经典的多智能体避障算法
    
    参考: van den Berg et al., "Reciprocal n-Body Collision Avoidance", ICRA 2011
    
    针对你的场景改进:
    1. 降低安全边距，更敢于接近行人
    2. 增大目标导向权重，更敢于推进
    3. 添加 LiDAR 静态障碍物过滤
    4. 添加 stuck recovery 机制
    """
    
    def __init__(self, cfg: ORCAConfig):
        self.cfg = cfg
        self.prev_vx = 0.0
        self.prev_vw = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        
        # Stuck detection
        self.velocity_history = []
        self.goal_dist_history = []
        self.is_recovery_mode = False
        self.recovery_counter = 0
        self.recovery_direction = 0.0
        
    def reset(self):
        self.prev_vx = 0.0
        self.prev_vw = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.velocity_history = []
        self.goal_dist_history = []
        self.is_recovery_mode = False
        self.recovery_counter = 0
        self.recovery_direction = 0.0
        
    def act(self, obs: np.ndarray) -> np.ndarray:
        """主决策函数"""
        # 解析观测
        lidar_norm = obs[:180]
        lidar_m = lidar_norm * self.cfg.lidar_max_range
        
        goal_dist = float(obs[182]) * 30.0
        goal_angle = float(obs[183]) * math.pi
        cur_v = float(obs[184]) * self.cfg.max_vel_x
        cur_w = float(obs[185]) * self.cfg.max_vel_theta
        
        # 解析行人信息 (obs[187:202])
        pedestrians = self._parse_pedestrians(obs, cur_v)
        
        # 更新历史
        self.velocity_history.append(cur_v)
        self.goal_dist_history.append(goal_dist)
        if len(self.velocity_history) > self.cfg.stuck_window:
            self.velocity_history.pop(0)
            self.goal_dist_history.pop(0)
        
        # 检测是否卡住
        is_stuck = self._detect_stuck()
        
        # Recovery 模式
        if self.is_recovery_mode:
            self.recovery_counter += 1
            if self.recovery_counter >= self.cfg.recovery_steps:
                self.is_recovery_mode = False
                self.recovery_counter = 0
            else:
                return self._execute_recovery(lidar_m, goal_angle)
        
        if is_stuck and not self.is_recovery_mode:
            self.is_recovery_mode = True
            self.recovery_counter = 0
            left_min = float(np.min(lidar_m[0:60]))
            right_min = float(np.min(lidar_m[120:180]))
            self.recovery_direction = 1.0 if left_min > right_min else -1.0
            return self._execute_recovery(lidar_m, goal_angle)
        
        # ORCA 规划
        action = self._orca_plan(lidar_m, goal_dist, goal_angle, cur_v, cur_w, pedestrians)
        
        self.prev_vx = action[0] * self.cfg.max_vel_x
        self.prev_vw = action[1] * self.cfg.max_vel_theta
        self.prev_action = action.copy()
        
        return action
    
    def _parse_pedestrians(self, obs: np.ndarray, cur_v: float) -> list:
        """解析行人观测"""
        pedestrians = []
        
        if obs.shape[0] < 202:
            return pedestrians
        
        ped_block = obs[187:202]
        
        for i in range(3):  # 最多3个行人
            base = i * 5
            rel_x = float(ped_block[base + 0]) * self.cfg.neighbor_dist
            rel_z = float(ped_block[base + 1]) * self.cfg.neighbor_dist
            ped_vx = float(ped_block[base + 2]) * 2.0
            ped_vz = float(ped_block[base + 3]) * 2.0
            radius = float(ped_block[base + 4])
            
            # 检查是否为有效行人
            if abs(rel_x) < 1e-6 and abs(rel_z) < 1e-6:
                continue
            
            # 计算相对速度
            v_rel_x = ped_vx - 0.0  # 机器人侧向速度为0
            v_rel_z = ped_vz - cur_v
            
            pedestrians.append({
                'position': np.array([rel_x, rel_z], dtype=np.float32),
                'velocity': np.array([ped_vx, ped_vz], dtype=np.float32),
                'rel_velocity': np.array([v_rel_x, v_rel_z], dtype=np.float32),
                'radius': radius if radius > 0.1 else self.cfg.pedestrian_radius
            })
        
        return pedestrians
    
    def _detect_stuck(self) -> bool:
        """检测是否陷入停滞"""
        if len(self.velocity_history) < self.cfg.stuck_window:
            return False
        
        avg_vel = np.mean(self.velocity_history)
        if avg_vel < self.cfg.stuck_velocity_threshold:
            return True
        
        if len(self.goal_dist_history) >= self.cfg.stuck_window:
            progress = self.goal_dist_history[0] - self.goal_dist_history[-1]
            if progress < 0.2:
                return True
        
        return False
    
    def _execute_recovery(self, lidar_m: np.ndarray, goal_angle: float) -> np.ndarray:
        """执行恢复策略"""
        front_min = float(np.min(lidar_m[75:105]))
        
        if front_min > 1.0:
            return np.array([0.5, 0.0], dtype=np.float32)
        
        if abs(goal_angle) < math.pi / 4:
            action_w = np.sign(goal_angle) * 0.3
        else:
            action_w = self.recovery_direction * self.cfg.recovery_vw
        
        return np.array([
            self.cfg.recovery_vx / self.cfg.max_vel_x,
            action_w / self.cfg.max_vel_theta
        ], dtype=np.float32)
    
    def _orca_plan(self, lidar_m: np.ndarray, goal_dist: float, goal_angle: float,
                   cur_v: float, cur_w: float, pedestrians: list) -> np.ndarray:
        """ORCA 核心: 速度采样 + 碰撞避免"""
        
        # 计算偏好速度 (指向目标)
        pref_speed = min(self.cfg.max_vel_x, goal_dist / self.cfg.time_horizon)
        if goal_dist < 2.0:
            pref_speed *= max(0.2, goal_dist / 2.0)
        
        pref_vx = pref_speed * math.sin(goal_angle)
        pref_vz = pref_speed * math.cos(goal_angle)
        pref_vel = np.array([pref_vx, pref_vz], dtype=np.float32)
        
        # 生成候选速度
        candidates = self._sample_velocities()
        
        best_score = -float('inf')
        best_vel = pref_vel.copy()
        
        for vel in candidates:
            # 检查是否与行人碰撞
            if self._check_pedestrian_collision(vel, pedestrians):
                continue
            
            # 检查是否与静态障碍物碰撞
            heading = math.atan2(vel[0], vel[1] + 1e-6)
            speed = float(np.linalg.norm(vel))
            speed_limit = self._static_speed_limit(lidar_m, heading)
            
            if speed > speed_limit + 0.05:
                continue
            
            # 评分
            score = self._score_velocity(vel, pref_vel, cur_v, cur_w)
            
            if score > best_score:
                best_score = score
                best_vel = vel.copy()
        
        # 转换为动作
        return self._velocity_to_action(best_vel, cur_v, cur_w)
    
    def _sample_velocities(self) -> list:
        """采样候选速度"""
        candidates = [np.zeros(2, dtype=np.float32)]
        
        speeds = np.linspace(0.1, self.cfg.max_vel_x, self.cfg.n_speed_samples)
        headings = np.linspace(-math.pi * 0.8, math.pi * 0.8, self.cfg.n_heading_samples)
        
        for s in speeds:
            for h in headings:
                vx = s * math.sin(h)
                vz = s * math.cos(h)
                candidates.append(np.array([vx, vz], dtype=np.float32))
        
        return candidates
    
    def _check_pedestrian_collision(self, vel: np.ndarray, pedestrians: list) -> bool:
        """检查速度是否会导致与行人碰撞 (ORCA 核心)"""
        for ped in pedestrians:
            p_rel = ped['position']
            v_rel = ped['rel_velocity']
            combined_r = self.cfg.robot_radius + ped['radius'] + self.cfg.safety_margin
            
            # 当前距离
            dist_now = float(np.linalg.norm(p_rel))
            if dist_now < combined_r:
                return True
            
            # 计算相对速度
            v_test_rel = vel - v_rel
            
            # 找到最近点的时间
            T = self.cfg.time_horizon
            vv = float(np.dot(v_test_rel, v_test_rel))
            
            if vv < 1e-8:
                t_star = 0.0
            else:
                t_star = -float(np.dot(p_rel, v_test_rel)) / vv
                t_star = np.clip(t_star, 0.0, T)
            
            # 最近距离
            closest = p_rel + t_star * v_test_rel
            d_min = float(np.linalg.norm(closest))
            
            if d_min < combined_r:
                return True
        
        return False
    
    def _static_speed_limit(self, lidar_m: np.ndarray, heading: float) -> float:
        """根据 LiDAR 计算静态障碍物限速"""
        center_idx = int(np.clip(90 + math.degrees(heading), 0, 179))
        half = 15
        
        left = max(0, center_idx - half)
        right = min(179, center_idx + half)
        sector = lidar_m[left:right + 1]
        
        if len(sector) == 0:
            return 0.0
        
        min_dist = float(np.min(sector))
        clearance = min_dist - self.cfg.robot_radius
        
        if clearance <= self.cfg.static_stop_dist:
            return 0.0
        
        if clearance <= self.cfg.static_slow_dist:
            alpha = (clearance - self.cfg.static_stop_dist) / (
                self.cfg.static_slow_dist - self.cfg.static_stop_dist + 1e-6
            )
            return alpha * self.cfg.max_vel_x
        
        return self.cfg.max_vel_x
    
    def _score_velocity(self, vel: np.ndarray, pref_vel: np.ndarray, 
                        cur_v: float, cur_w: float) -> float:
        """评估候选速度"""
        speed = float(np.linalg.norm(vel))
        heading = math.atan2(vel[0], vel[1] + 1e-6)
        
        # 1. 目标距离得分
        dist_to_pref = float(np.linalg.norm(vel - pref_vel))
        goal_score = -self.cfg.goal_weight * dist_to_pref
        
        # 2. 速度奖励
        speed_score = self.cfg.speed_weight * speed
        
        # 3. 朝向奖励 (前进方向)
        heading_score = self.cfg.heading_weight * math.cos(heading)
        
        # 4. 平滑惩罚
        prev_vel = np.array([
            self.prev_vx * math.sin(self.prev_vw * 0.1),
            self.prev_vx * math.cos(self.prev_vw * 0.1)
        ], dtype=np.float32)
        smooth_penalty = self.cfg.smooth_weight * float(np.linalg.norm(vel - prev_vel))
        
        # 5. 前进奖励
        forward_bonus = 0.5 if vel[1] > 0.1 else 0.0
        
        return goal_score + speed_score + heading_score - smooth_penalty + forward_bonus
    
    def _velocity_to_action(self, vel: np.ndarray, cur_v: float, cur_w: float) -> np.ndarray:
        """将速度转换为动作"""
        speed = float(np.linalg.norm(vel))
        heading = math.atan2(vel[0], vel[1] + 1e-6)
        
        # 前进速度
        forward_factor = max(0.0, math.cos(heading))
        target_v = speed * forward_factor
        
        # 角速度
        target_w = np.clip(heading / 0.5, -self.cfg.max_vel_theta, self.cfg.max_vel_theta)
        
        # 归一化
        v_norm = target_v / self.cfg.max_vel_x
        w_norm = target_w / self.cfg.max_vel_theta
        
        # 平滑
        v_norm = np.clip(v_norm, -1.0, 1.0)
        w_norm = np.clip(w_norm, -1.0, 1.0)
        
        # 动作平滑
        v_norm = np.clip(v_norm, self.prev_action[0] - 0.2, self.prev_action[0] + 0.2)
        w_norm = np.clip(w_norm, self.prev_action[1] - 0.3, self.prev_action[1] + 0.3)
        
        return np.array([v_norm, w_norm], dtype=np.float32)


# =============================================================================
# APF (Artificial Potential Field) - Khatib, 1986 标准实现 + 逃逸机制
# =============================================================================
@dataclass
class APFConfig:
    """APF 算法配置 - 改进版包含逃逸机制"""
    
    # 势场参数 - 斥力应强于吸引力以确保安全
    attractive_gain: float = 1.0    # 吸引力增益
    repulsive_gain: float = 2.0     # 斥力增益 (提高以增强避障)
    
    # 障碍物参数
    robot_radius: float = 0.25
    safe_distance: float = 2.5      # 斥力影响距离
    
    # 速度限制
    max_vel_x: float = 1.0
    max_vel_theta: float = 1.5
    
    # 控制参数
    kp_linear: float = 0.8          # 线速度比例
    kp_angular: float = 2.0         # 角速度比例
    
    # 逃逸机制参数
    escape_threshold: float = 0.3   # 速度低于此值认为可能卡住
    escape_window: int = 15         # 检测窗口大小
    escape_gain: float = 0.5        # 逃逸扰动强度


class APFPolicy:
    """
    APF 策略实现 - 基于 Khatib, 1986 论文 + 逃逸机制
    
    核心思想:
    1. 目标产生吸引力
    2. 障碍物产生斥力
    3. 合力决定运动方向
    4. 检测局部最小值并添加逃逸扰动
    """
    
    def __init__(self, cfg: APFConfig):
        self.cfg = cfg
        self.velocity_history = []  # 历史速度记录
        self.escape_mode = False    # 逃逸模式标志
        self.escape_direction = 0.0 # 逃逸方向
        self.escape_counter = 0     # 逃逸计数器
        
    def reset(self):
        self.velocity_history = []
        self.escape_mode = False
        self.escape_direction = 0.0
        self.escape_counter = 0
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """根据观测选择动作 - APF 只使用 LiDAR 和目标信息，不使用行人数据"""
        # 解析观测 (只用前 187 维，与 PPO 保持公平对比)
        lidar_norm = obs[:180]
        goal_dist = float(obs[182]) * 30.0  # maxScenarioSize = 30m
        goal_angle = float(obs[183]) * math.pi
        cur_v = float(obs[184]) * self.cfg.max_vel_x
        
        lidar_m = lidar_norm * 10.0  # maxDistance = 10m
        
        # 检测是否卡住
        self.velocity_history.append(cur_v)
        if len(self.velocity_history) > self.cfg.escape_window:
            self.velocity_history.pop(0)
        
        # 判断是否进入逃逸模式
        if len(self.velocity_history) >= self.cfg.escape_window:
            avg_vel = np.mean(self.velocity_history)
            if avg_vel < self.cfg.escape_threshold and not self.escape_mode:
                self.escape_mode = True
                self.escape_counter = 0
                # 选择逃逸方向 - 选择空间更大的一侧
                left_min = float(np.min(lidar_m[0:60]))
                right_min = float(np.min(lidar_m[120:180]))
                self.escape_direction = 1.0 if left_min > right_min else -1.0
        
        # 逃逸模式处理
        if self.escape_mode:
            self.escape_counter += 1
            if self.escape_counter > 20:  # 逃逸20步后恢复正常
                self.escape_mode = False
                self.velocity_history = []
            else:
                # 执行逃逸动作
                action_v = 0.3
                action_w = self.escape_direction * 0.8
                return np.array([action_v, action_w], dtype=np.float32)
        
        # 计算吸引力
        f_att = self._compute_attractive_force(goal_dist, goal_angle)
        
        # 计算斥力
        f_rep = self._compute_repulsive_force(lidar_m)
        
        # 合力
        f_total = f_att + f_rep
        
        # 计算期望速度
        vx = f_total[1] * self.cfg.kp_linear  # 前进方向
        vtheta = f_total[0] * self.cfg.kp_angular  # 侧向转为角速度
        
        # 限制速度
        vx = np.clip(vx, 0.0, self.cfg.max_vel_x)
        vtheta = np.clip(vtheta, -self.cfg.max_vel_theta, self.cfg.max_vel_theta)
        
        # 转换为归一化动作
        action_v = vx / self.cfg.max_vel_x
        action_w = vtheta / self.cfg.max_vel_theta
        
        return np.array([action_v, action_w], dtype=np.float32)
    
    def _compute_attractive_force(self, goal_dist: float, goal_angle: float) -> np.ndarray:
        """计算目标产生的吸引力"""
        # 目标在局部坐标系中的位置
        goal_x = goal_dist * math.sin(goal_angle)
        goal_z = goal_dist * math.cos(goal_angle)
        
        # 吸引力方向：指向目标
        if goal_dist > 0.01:
            f_att = self.cfg.attractive_gain * np.array([goal_x, goal_z]) / goal_dist
        else:
            f_att = np.array([0.0, 0.0])
        
        return f_att
    
    def _compute_repulsive_force(self, lidar_m: np.ndarray) -> np.ndarray:
        """计算障碍物产生的斥力"""
        f_rep = np.array([0.0, 0.0])
        
        for i, dist in enumerate(lidar_m):
            if dist < self.cfg.safe_distance:
                # 障碍物方向角度
                angle = math.radians(i - 90)  # 90度为前方
                
                # 障碍物在局部坐标系中的位置
                obs_x = dist * math.sin(angle)
                obs_z = dist * math.cos(angle)
                
                # 斥力大小：距离越近斥力越大
                if dist < self.cfg.robot_radius:
                    dist = self.cfg.robot_radius
                
                # 斥力公式
                magnitude = self.cfg.repulsive_gain * (
                    1.0 / dist - 1.0 / self.cfg.safe_distance
                ) * (1.0 / dist ** 2)
                
                # 斥力方向：远离障碍物
                direction = np.array([-obs_x, -obs_z])
                if np.linalg.norm(direction) > 0.01:
                    direction = direction / np.linalg.norm(direction)
                
                f_rep += magnitude * direction
        
        return f_rep


# =============================================================================
# 简化的 TEB (Timed Elastic Band) - 基于轨迹优化
# =============================================================================
@dataclass
class TEBConfig:
    """简化 TEB 配置"""
    
    robot_radius: float = 0.25
    max_vel_x: float = 1.0
    max_vel_theta: float = 1.5
    
    # 优化参数 - 障碍物代价要足够大
    weight_goal: float = 1.0
    weight_obstacle: float = 10.0    # 大幅提高障碍物代价
    weight_velocity: float = 0.5
    
    # 预测时域
    horizon: float = 3.0
    dt: float = 0.1
    
    # 采样数
    n_samples: int = 100


class TEBPolicy:
    """
    简化 TEB 策略 - 基于采样的轨迹优化
    
    核心思想:
    1. 采样多条候选轨迹
    2. 对每条轨迹计算代价（目标 + 障碍物 + 速度）
    3. 选择代价最小的轨迹
    """
    
    def __init__(self, cfg: TEBConfig):
        self.cfg = cfg
        
    def reset(self):
        pass
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """根据观测选择动作 - TEB 只使用 LiDAR 和目标信息，不使用行人数据"""
        # 解析观测 (只用前 187 维，与 PPO 保持公平对比)
        lidar_norm = obs[:180]
        goal_dist = float(obs[182]) * 30.0  # maxScenarioSize = 30m
        goal_angle = float(obs[183]) * math.pi
        
        lidar_m = lidar_norm * 10.0  # maxDistance = 10m
        
        best_cost = float('inf')
        best_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # 采样候选动作
        for _ in range(self.cfg.n_samples):
            # 随机采样速度
            vx = np.random.uniform(0, self.cfg.max_vel_x)
            vtheta = np.random.uniform(-self.cfg.max_vel_theta, self.cfg.max_vel_theta)
            
            # 模拟轨迹
            trajectory = self._simulate(vx, vtheta)
            
            # 计算代价
            cost = self._compute_cost(trajectory, vx, vtheta, goal_dist, goal_angle, lidar_m)
            
            if cost < best_cost:
                best_cost = cost
                best_action = np.array([vx / self.cfg.max_vel_x, 
                                        vtheta / self.cfg.max_vel_theta], 
                                       dtype=np.float32)
        
        return best_action
    
    def _simulate(self, vx: float, vtheta: float) -> List[Tuple[float, float]]:
        """模拟轨迹"""
        trajectory = []
        x, y, theta = 0.0, 0.0, 0.0
        
        for _ in range(int(self.cfg.horizon / self.cfg.dt)):
            x += vx * self.cfg.dt * math.cos(theta)
            y += vx * self.cfg.dt * math.sin(theta)
            theta += vtheta * self.cfg.dt
            trajectory.append((x, y, theta))
        
        return trajectory
    
    def _compute_cost(self, trajectory: List[Tuple[float, float]], 
                      vx: float, vtheta: float,
                      goal_dist: float, goal_angle: float,
                      lidar_m: np.ndarray) -> float:
        """计算轨迹代价"""
        if not trajectory:
            return float('inf')
        
        # 目标代价
        end_x, end_y, end_theta = trajectory[-1]
        goal_x = goal_dist * math.sin(goal_angle)
        goal_y = goal_dist * math.cos(goal_angle)
        goal_cost = self.cfg.weight_goal * math.sqrt((end_x - goal_x)**2 + (end_y - goal_y)**2)
        
        # 障碍物代价
        obs_cost = 0.0
        collision = False
        for x, y, theta in trajectory:
            # 局部坐标系: x=侧向, y=前方
            # LiDAR: 索引 90=前方, 0=左侧, 179=右侧
            point_angle = math.atan2(x, y)  # 侧向角度
            lidar_idx = int(90 - math.degrees(point_angle))
            lidar_idx = max(0, min(179, lidar_idx))
            lidar_dist = lidar_m[lidar_idx]
            point_dist = math.sqrt(x**2 + y**2)
            
            # 计算到障碍物的距离
            dist_to_obs = lidar_dist - point_dist
            
            if dist_to_obs < self.cfg.robot_radius:
                collision = True
                break  # 碰撞，直接终止
            elif dist_to_obs < 1.0:  # 安全距离内
                obs_cost += self.cfg.weight_obstacle * (1.0 - dist_to_obs) ** 2
        
        # 碰撞时返回极大代价
        if collision:
            return float('inf')
        
        # 速度代价：保持前进
        vel_cost = -self.cfg.weight_velocity * vx
        
        return goal_cost + obs_cost + vel_cost


# =============================================================================
# VFH+ (Vector Field Histogram+) - Ulrich & Borenstein, 1998
# =============================================================================
@dataclass
class VFHConfig:
    """VFH+ 算法配置"""
    
    # 机器人参数
    robot_radius: float = 0.25
    max_vel_x: float = 1.0
    max_vel_theta: float = 1.5
    
    # 直方图参数
    alpha: float = 5.0           # 扇区角度分辨率 (度)
    threshold_low: float = 0.3   # 二值化低阈值
    threshold_high: float = 0.8  # 二值化高阈值
    
    # 代价函数权重
    weight_goal: float = 5.0     # 目标方向权重
    weight_smooth: float = 2.0   # 平滑性权重
    weight_prev: float = 3.0     # 与上一方向一致性权重
    
    # 安全距离
    safety_dist: float = 0.5     # 安全距离
    min_turn_radius: float = 0.3 # 最小转弯半径


class VFHPolicy:
    """
    VFH+ 策略实现 - 基于 Ulrich & Borenstein, 1998 论文
    
    核心思想:
    1. 将LiDAR数据转换为极坐标直方图
    2. 对直方图进行平滑和二值化处理
    3. 在可行方向中选择最优朝向
    """
    
    def __init__(self, cfg: VFHConfig):
        self.cfg = cfg
        self.prev_direction = 0.0  # 上一次选择的方向
        self.stuck_counter = 0     # 卡住计数器
        
    def reset(self):
        self.prev_direction = 0.0
        self.stuck_counter = 0
        
    def act(self, obs: np.ndarray) -> np.ndarray:
        """根据观测选择动作"""
        # 解析观测
        lidar_norm = obs[:180]
        goal_dist = float(obs[182]) * 30.0
        goal_angle = float(obs[183]) * math.pi  # 目标角度
        
        lidar_m = lidar_norm * 10.0
        
        # 1. 构建极坐标直方图
        histogram = self._build_histogram(lidar_m)
        
        # 2. 平滑直方图
        smoothed = self._smooth_histogram(histogram)
        
        # 3. 二值化
        binary = self._binarize_histogram(smoothed)
        
        # 4. 找到可行方向
        candidate_dirs = self._find_free_directions(binary)
        
        if not candidate_dirs:
            # 没有可行方向，原地旋转
            self.stuck_counter += 1
            return self._escape_behavior(lidar_m, goal_angle)
        
        # 5. 选择最优方向
        best_dir = self._select_direction(candidate_dirs, goal_angle)
        
        self.prev_direction = best_dir
        self.stuck_counter = 0
        
        # 6. 计算速度
        vx, vtheta = self._compute_velocity(best_dir, goal_dist, lidar_m)
        
        # 归一化
        action_v = vx / self.cfg.max_vel_x
        action_w = vtheta / self.cfg.max_vel_theta
        
        return np.array([action_v, action_w], dtype=np.float32)
    
    def _build_histogram(self, lidar_m: np.ndarray) -> np.ndarray:
        """构建极坐标直方图 - 每个扇区的障碍物密度"""
        # 180个LiDAR点，每个点代表1度
        # 每个扇区包含多个LiDAR点，计算障碍物密度
        sector_size = max(1, int(self.cfg.alpha))  # 每个扇区的LiDAR点数
        n_sectors = 180 // sector_size
        
        histogram = np.zeros(n_sectors)
        
        for i in range(n_sectors):
            start_idx = i * sector_size
            end_idx = min((i + 1) * sector_size, 180)
            sector_data = lidar_m[start_idx:end_idx]
            
            # 计算该扇区的障碍物密度
            # 距离越近，密度越高
            for d in sector_data:
                if d < 5.0:  # 只考虑5米内的障碍
                    # 密度与距离成反比
                    density = max(0, (5.0 - d) / 5.0) ** 2
                    histogram[i] = max(histogram[i], density)
        
        return histogram
    
    def _smooth_histogram(self, histogram: np.ndarray) -> np.ndarray:
        """平滑直方图 - 低通滤波"""
        n = len(histogram)
        smoothed = np.zeros(n)
        
        kernel_size = 3
        for i in range(n):
            total = 0.0
            count = 0
            for k in range(-kernel_size, kernel_size + 1):
                idx = (i + k) % n
                weight = 1.0 - abs(k) / (kernel_size + 1)
                total += histogram[idx] * weight
                count += weight
            smoothed[i] = total / max(count, 1e-6)
        
        return smoothed
    
    def _binarize_histogram(self, smoothed: np.ndarray) -> np.ndarray:
        """二值化直方图 - 阈值处理"""
        binary = np.zeros_like(smoothed)
        
        for i in range(len(smoothed)):
            if smoothed[i] > self.cfg.threshold_high:
                binary[i] = 1.0  # 阻塞
            elif smoothed[i] < self.cfg.threshold_low:
                binary[i] = 0.0  # 空闲
            else:
                # 滞后阈值 - 保持之前状态
                binary[i] = binary[i-1] if i > 0 else 0.0
        
        return binary
    
    def _find_free_directions(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """找到所有连续的空闲方向区间"""
        n = len(binary)
        free_regions = []
        
        # 找到所有空闲区间的起始和结束
        start = None
        for i in range(n * 2):  # 环形遍历
            idx = i % n
            if binary[idx] < 0.5:  # 空闲
                if start is None:
                    start = idx
            else:
                if start is not None:
                    end = (i - 1) % n
                    if start != end:
                        free_regions.append((start, end))
                    start = None
        
        return free_regions
    
    def _select_direction(self, candidates: List[Tuple[int, int]], goal_angle: float) -> float:
        """从候选方向中选择最优方向"""
        # 将目标角度转换为LiDAR索引
        # goal_angle: 正=目标在左侧, 负=目标在右侧
        # LiDAR: 0=左侧(-90°), 90=前方(0°), 179=右侧(+89°)
        goal_idx = int(90 - math.degrees(goal_angle))
        goal_idx = max(0, min(179, goal_idx))
        
        # 目标方向所在扇区
        sector_size = max(1, int(self.cfg.alpha))
        goal_sector = goal_idx // sector_size
        
        best_score = -float('inf')
        best_dir = 0
        
        n = int(180 / sector_size)
        
        for start, end in candidates:
            # 计算区间中心方向
            if start <= end:
                center = (start + end) // 2
            else:
                # 跨越边界的区间
                center = ((start + end + n) // 2) % n
            
            # 目标代价: 方向与目标越接近越好
            goal_cost = min(abs(center - goal_sector), 
                           n - abs(center - goal_sector))
            goal_score = -self.cfg.weight_goal * goal_cost
            
            # 平滑代价: 与上一方向越接近越好
            prev_cost = min(abs(center - self.prev_direction), 
                           n - abs(center - self.prev_direction))
            smooth_score = -self.cfg.weight_prev * prev_cost
            
            # 区间宽度奖励: 越宽越安全
            width = end - start + 1 if start <= end else n - start + end + 1
            width_score = self.cfg.weight_smooth * (width / n)
            
            total_score = goal_score + smooth_score + width_score
            
            if total_score > best_score:
                best_score = total_score
                best_dir = center
        
        # 转换为角度
        angle = (best_dir * self.cfg.alpha - 90) * math.pi / 180
        return angle
    
    def _compute_velocity(self, direction: float, goal_dist: float, 
                          lidar_m: np.ndarray) -> Tuple[float, float]:
        """计算线速度和角速度"""
        # 前方LiDAR最小距离
        front_sector = lidar_m[75:105]  # 前方30度
        front_min = float(np.min(front_sector)) if len(front_sector) > 0 else 10.0
        
        # 根据前方距离调整速度
        if front_min < self.cfg.safety_dist:
            speed_factor = front_min / self.cfg.safety_dist
        else:
            speed_factor = 1.0
        
        # 接近目标时减速
        if goal_dist < 2.0:
            speed_factor *= max(0.3, goal_dist / 2.0)
        
        # 线速度: 前向分量
        vx = self.cfg.max_vel_x * speed_factor * max(0, math.cos(direction))
        
        # 角速度: 转向目标方向
        vtheta = np.clip(direction * 2.0, -self.cfg.max_vel_theta, self.cfg.max_vel_theta)
        
        return vx, vtheta
    
    def _escape_behavior(self, lidar_m: np.ndarray, goal_angle: float) -> np.ndarray:
        """卡住时的逃逸行为"""
        # 检查左右空间，选择空间更大的一侧转向
        left_min = float(np.min(lidar_m[0:60]))
        right_min = float(np.min(lidar_m[120:180]))
        
        if left_min > right_min:
            # 左侧空间更大，左转
            return np.array([0.1, 0.8], dtype=np.float32)
        else:
            # 右侧空间更大，右转
            return np.array([0.1, -0.8], dtype=np.float32)


# =============================================================================
# 停滞检测
# =============================================================================
@dataclass
class StagnationConfig:
    velocity_threshold: float = 0.05
    angular_threshold: float = 0.3
    min_stagnation_steps: int = 3


def detect_stagnation(velocity: float, angular: float, cfg: StagnationConfig) -> bool:
    return velocity < cfg.velocity_threshold or (
        velocity < cfg.velocity_threshold * 2 and abs(angular) > cfg.angular_threshold
    )


# =============================================================================
# 统一评估框架
# =============================================================================
def run_episode(env: UnityNavEnv, policy, policy_name: str = "Policy",
                stagnation_cfg: StagnationConfig = None) -> Dict:
    if stagnation_cfg is None:
        stagnation_cfg = StagnationConfig()
    
    obs, info = env.reset()
    if hasattr(policy, 'reset'):
        policy.reset()

    done = False
    ep_ret = 0.0
    ep_len = 0
    last_info = {}
    
    min_lidar_list = []
    velocity_list = []
    stagnation_list = []

    while not done:
        action = policy.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        ep_ret += reward
        ep_len += 1
        last_info = info
        
        min_lidar_list.append(float(np.min(obs[:180]) * 10.0))
        velocity_list.append(float(obs[184]))
        
        is_stagnant = detect_stagnation(float(obs[184]), float(action[1]), stagnation_cfg)
        stagnation_list.append(is_stagnant)

    stagnation_steps = sum(stagnation_list)
    stagnation_rate = stagnation_steps / ep_len if ep_len > 0 else 0.0
    
    result = {
        "policy": policy_name,
        "return": ep_ret,
        "length": ep_len,
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "timeout": bool(last_info.get("timeout", False)),
        "final_goal_dist": float(last_info.get("goal_dist", np.nan)),
        "min_lidar_mean": float(np.mean(min_lidar_list)) if min_lidar_list else 0.0,
        "min_lidar_min": float(np.min(min_lidar_list)) if min_lidar_list else 0.0,
        "velocity_mean": float(np.mean(velocity_list)) if velocity_list else 0.0,
        "stagnation_steps": stagnation_steps,
        "stagnation_rate": stagnation_rate,
    }
    return result


def evaluate_policy(env_cfg: EnvConfig, policy, policy_name: str, 
                    n_episodes: int = 50) -> Dict:
    env = UnityNavEnv(env_cfg)
    
    results = []
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {policy_name}")
    print(f"{'='*60}")
    
    for i in range(n_episodes):
        result = run_episode(env, policy, policy_name)
        results.append(result)
        
        status = "SUCCESS" if result["success"] else (
            "COLLISION" if result["collision"] else "TIMEOUT")
        print(f"[{policy_name}][{i+1:03d}] len={result['length']:3d} | "
              f"dist={result['final_goal_dist']:.2f}m | {status}")

    env.close()
    eval_time = time.time() - start_time

    summary = {
        "policy": policy_name,
        "n_episodes": n_episodes,
        "success_rate": np.mean([r["success"] for r in results]),
        "collision_rate": np.mean([r["collision"] for r in results]),
        "timeout_rate": np.mean([r["timeout"] for r in results]),
        "avg_length": np.mean([r["length"] for r in results]),
        "std_length": np.std([r["length"] for r in results]),
        "avg_return": np.mean([r["return"] for r in results]),
        "std_return": np.std([r["return"] for r in results]),
        "avg_final_goal_dist": np.nanmean([r["final_goal_dist"] for r in results]),
        "avg_min_lidar": np.mean([r["min_lidar_mean"] for r in results]),
        "min_lidar_min": np.min([r["min_lidar_min"] for r in results]),
        "avg_velocity": np.mean([r["velocity_mean"] for r in results]),
        "avg_stagnation_rate": np.mean([r["stagnation_rate"] for r in results]),
        "std_stagnation_rate": np.std([r["stagnation_rate"] for r in results]),
        "eval_time": eval_time,
    }

    print(f"\n{policy_name} Summary:")
    print(f"  success_rate      : {summary['success_rate']:.3f}")
    print(f"  collision_rate    : {summary['collision_rate']:.3f}")
    print(f"  timeout_rate      : {summary['timeout_rate']:.3f}")
    print(f"  avg_length        : {summary['avg_length']:.1f} ± {summary['std_length']:.1f}")
    print(f"  avg_return        : {summary['avg_return']:.3f} ± {summary['std_return']:.3f}")
    print(f"  avg_final_dist    : {summary['avg_final_goal_dist']:.2f} m")
    print(f"  avg_stagnation_rate: {summary['avg_stagnation_rate']:.3f} (冻结问题指标)")

    return summary


def compare_baselines(env_cfg: EnvConfig, n_episodes: int = 50,
                      save_dir: str = "./results") -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义所有策略 - 经典传统方法
    policies = {
        "ORCA": (ORCAPolicy(ORCAConfig()), ORCAConfig()),           # 经典多智能体避障
        "DWA_V2": (DWAPolicyV2(DWAConfig()), DWAConfig()),          # 改进版 DWA
        "APF": (APFPolicy(APFConfig()), APFConfig()),               # 人工势场
        "VFH+": (VFHPolicy(VFHConfig()), VFHConfig()),              # 向量场直方图
        "TEB": (TEBPolicy(TEBConfig()), TEBConfig()),               # 时间弹性带
    }
    
    results = {}
    
    for name, (policy, cfg) in policies.items():
        print(f"\n{'#'*60}")
        print(f"# Evaluating {name}")
        print(f"{'#'*60}")
        summary = evaluate_policy(env_cfg, policy, name, n_episodes)
        results[name] = summary
    
    # 生成对比表格
    print(f"\n{'='*100}")
    print("BASELINE COMPARISON SUMMARY (V2)")
    print(f"{'='*100}")
    print(f"{'Policy':<12} {'Success':>8} {'Collision':>10} {'Timeout':>8} "
          f"{'Avg Len':>10} {'Avg Dist':>10} {'Stagnation':>12}")
    print(f"{'-'*100}")
    
    for name, summary in results.items():
        print(f"{name:<12} {summary['success_rate']:>8.3f} "
              f"{summary['collision_rate']:>10.3f} "
              f"{summary['timeout_rate']:>8.3f} "
              f"{summary['avg_length']:>10.1f} "
              f"{summary['avg_final_goal_dist']:>10.2f} "
              f"{summary['avg_stagnation_rate']:>12.3f}")
    
    print(f"{'='*100}")
    print("注: Stagnation Rate = 停滞时间 / 总运行时间，反映\"冻结问题\"严重程度\n")
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    json_path = os.path.join(save_dir, f"baseline_comparison_v2_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    csv_path = os.path.join(save_dir, f"baseline_comparison_v2_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Policy', 'Success Rate', 'Collision Rate', 'Timeout Rate',
                        'Avg Length', 'Std Length', 'Avg Return', 'Std Return',
                        'Avg Final Dist', 'Avg Min LiDAR', 'Avg Velocity', 
                        'Stagnation Rate', 'Std Stagnation Rate'])
        for name, s in results.items():
            writer.writerow([name, s['success_rate'], s['collision_rate'],
                           s['timeout_rate'], s['avg_length'], s['std_length'],
                           s['avg_return'], s['std_return'], s['avg_final_goal_dist'],
                           s['avg_min_lidar'], s['avg_velocity'],
                           s['avg_stagnation_rate'], s['std_stagnation_rate']])
    print(f"Results saved to: {csv_path}")
    
    return results


# =============================================================================
# 主函数
# =============================================================================
if __name__ == "__main__":
    env_cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\Builds_202\Project_1.exe",
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=202,
        lidar_dim=180,
        reach_goal_radius=0.5,
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
    
    results = compare_baselines(env_cfg, n_episodes=50, save_dir="./results")

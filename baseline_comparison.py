"""
基线算法对比评估模块

包含:
1. DWA (Dynamic Window Approach) - 基于规则的局部避障算法
2. ORCA (Optimal Reciprocal Collision Avoidance) - 多智能体避障算法
3. 统一对比评估框架

用于与 PPO 强化学习方法进行对比实验。
"""
import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from unity_env import UnityNavEnv, EnvConfig


# =============================================================================
# DWA (Dynamic Window Approach)
# =============================================================================
@dataclass
class DWAConfig:
    """DWA 算法配置"""
    
    # 机器人控制上限（与 Unity 中 RobotController 的速度上限一致）
    v_max: float = 1.0          # 最大线速度
    w_max: float = 1.0          # 最大角速度

    # 采样参数
    n_v: int = 11               # 线速度采样数
    n_w: int = 21               # 角速度采样数

    # 预测时域
    horizon_steps: int = 15     # 预测步数
    dt: float = 0.1             # 时间步长

    # 评分权重
    heading_weight: float = 2.0     # 朝向目标权重
    clearance_weight: float = 3.0   # 避障安全权重
    velocity_weight: float = 0.5    # 速度权重
    goal_weight: float = 1.5        # 目标距离权重

    # 安全参数
    robot_radius: float = 0.25      # 机器人半径
    brake_margin: float = 0.1       # 刹车安全余量
    hard_clearance: float = 0.15    # 硬性安全距离阈值

    # 动作平滑
    max_delta_v: float = 0.4        # 线速度最大变化
    max_delta_w: float = 0.5        # 角速度最大变化


class DWAPolicy:
    """DWA 策略实现"""
    
    def __init__(self, cfg: DWAConfig):
        self.cfg = cfg
        self.prev_action = np.zeros(2, dtype=np.float32)
        
    def reset(self):
        """重置策略状态"""
        self.prev_action[:] = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """根据观测选择动作"""
        # 解析观测
        lidar_norm = obs[:180]
        goal_dist = obs[182] * 30.0
        goal_angle = obs[183] * math.pi
        cur_v = float(obs[184])
        cur_w = float(obs[185])
        
        # LiDAR 归一化值转米
        lidar_m = lidar_norm * 10.0

        # 生成候选速度
        v_candidates = np.linspace(
            max(0.0, cur_v - self.cfg.max_delta_v),  # 不后退
            min(self.cfg.v_max, cur_v + self.cfg.max_delta_v),
            self.cfg.n_v
        )
        w_candidates = np.linspace(
            max(-self.cfg.w_max, cur_w - self.cfg.max_delta_w),
            min(self.cfg.w_max, cur_w + self.cfg.max_delta_w),
            self.cfg.n_w
        )

        # 遍历所有候选，找最优
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

    def _score_action(self, v: float, w: float, goal_angle: float, 
                      goal_dist: float, lidar_m: np.ndarray) -> float:
        """评估候选动作的得分"""
        # 1) heading score
        pred_heading = w * self.cfg.horizon_steps * self.cfg.dt
        heading_err = abs(self._wrap_to_pi(goal_angle - pred_heading))
        heading_score = 1.0 - min(heading_err / math.pi, 1.0)

        # 2) clearance score
        clearance = self._estimate_clearance(v, w, lidar_m)
        if clearance < self.cfg.hard_clearance:
            return -1e6
        
        clearance_score = np.tanh(clearance / 2.0)

        # 3) velocity score
        velocity_score = v / self.cfg.v_max

        # 4) goal score
        if goal_dist < 2.0:
            # 近目标时，强调朝向
            goal_score = 1.0 - min(goal_dist / 2.0, 1.0)
        else:
            goal_score = np.tanh(goal_dist / 15.0)

        # 综合得分
        score = (
            self.cfg.heading_weight * heading_score +
            self.cfg.clearance_weight * clearance_score +
            self.cfg.velocity_weight * velocity_score +
            self.cfg.goal_weight * goal_score
        )

        # 惩罚频繁转向
        score -= 0.05 * abs(w)
        
        # 前进奖励
        if v > 0.2:
            score += 0.15
            
        return float(score)

    def _estimate_clearance(self, v: float, w: float, lidar_m: np.ndarray) -> float:
        """估计当前速度方向上的安全余量"""
        center_deg = 90
        
        steer_offset = int(np.clip(w, -1.0, 1.0) * 25)
        center = center_deg + steer_offset

        half_width = 20 if abs(w) < 0.3 else 30
        left = max(0, center - half_width)
        right = min(179, center + half_width)

        sector = lidar_m[left:right + 1]
        if len(sector) == 0:
            return 0.0

        raw_clearance = float(np.min(sector))
        clearance = raw_clearance - self.cfg.robot_radius - self.cfg.brake_margin

        if v > 0:
            clearance -= v * 0.15
            
        return clearance

    @staticmethod
    def _wrap_to_pi(x: float) -> float:
        """将角度归一化到 [-π, π]"""
        return (x + math.pi) % (2 * math.pi) - math.pi


# =============================================================================
# ORCA (Optimal Reciprocal Collision Avoidance) - 简化版
# =============================================================================
@dataclass
class ORCAConfig:
    """ORCA 算法配置（简化版）"""
    
    v_max: float = 1.0
    w_max: float = 1.0
    robot_radius: float = 0.25
    time_horizon: float = 2.0      # 避障预测时间
    safety_margin: float = 0.3     # 安全边距
    
    # 目标吸引力
    goal_gain: float = 1.2
    
    # 动作平滑
    max_delta_v: float = 0.4
    max_delta_w: float = 0.5


class ORCAPolicy:
    """
    ORCA 策略实现（简化版）
    
    由于我们只能观测到 LiDAR 而非其他智能体的精确位置和速度，
    这里实现一个基于 LiDAR 的简化 ORCA 变体。
    """
    
    def __init__(self, cfg: ORCAConfig):
        self.cfg = cfg
        self.prev_action = np.zeros(2, dtype=np.float32)
        
    def reset(self):
        self.prev_action[:] = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """根据观测选择动作"""
        lidar_norm = obs[:180]
        goal_dist = obs[182] * 30.0
        goal_angle = obs[183] * math.pi
        cur_v = float(obs[184])
        cur_w = float(obs[185])
        
        lidar_m = lidar_norm * 10.0

        # 计算目标方向的速度偏好
        preferred_v = self._compute_preferred_velocity(goal_angle, goal_dist)
        
        # 根据 LiDAR 信息调整速度，避免碰撞
        safe_v = self._adjust_for_obstacles(preferred_v, lidar_m, cur_v, cur_w)
        
        # 动作平滑
        v = np.clip(safe_v[0], 
                    cur_v - self.cfg.max_delta_v, 
                    cur_v + self.cfg.max_delta_v)
        v = np.clip(v, 0.0, self.cfg.v_max)
        
        w = np.clip(safe_v[1],
                    cur_w - self.cfg.max_delta_w,
                    cur_w + self.cfg.max_delta_w)
        w = np.clip(w, -self.cfg.w_max, self.cfg.w_max)
        
        action = np.array([v, w], dtype=np.float32)
        self.prev_action = action.copy()
        return action

    def _compute_preferred_velocity(self, goal_angle: float, goal_dist: float) -> np.ndarray:
        """计算朝向目标的偏好速度"""
        # 角速度：减小目标角度误差
        w = -np.sign(goal_angle) * min(abs(goal_angle), self.cfg.w_max) * self.cfg.goal_gain
        
        # 线速度：角度小时前进快，角度大时减速转向
        if abs(goal_angle) < 0.3:
            v = self.cfg.v_max
        elif abs(goal_angle) < 1.0:
            v = self.cfg.v_max * (1.0 - abs(goal_angle))
        else:
            v = 0.0
        
        # 近目标时减速
        if goal_dist < 2.0:
            v *= min(goal_dist / 2.0 + 0.3, 1.0)
        
        return np.array([v, w])

    def _adjust_for_obstacles(self, preferred_v: np.ndarray, lidar_m: np.ndarray,
                               cur_v: float, cur_w: float) -> np.ndarray:
        """根据障碍物调整速度"""
        v, w = preferred_v
        
        # 检查前方障碍物
        front_sector = lidar_m[75:105]  # 前方 30 度范围
        front_min = float(np.min(front_sector)) if len(front_sector) > 0 else 10.0
        
        # 左右障碍物
        left_sector = lidar_m[0:60]
        right_sector = lidar_m[120:180]
        left_min = float(np.min(left_sector)) if len(left_sector) > 0 else 10.0
        right_min = float(np.min(right_sector)) if len(right_sector) > 0 else 10.0
        
        safe_dist = self.cfg.robot_radius + self.cfg.safety_margin
        
        # 前方有障碍物
        if front_min < safe_dist + 1.0:
            # 减速
            v = min(v, max(0, front_min - safe_dist) * 0.5)
            
            # 选择转向方向
            if left_min > right_min:
                w = 0.5 + 0.3 * (1.0 - front_min / (safe_dist + 1.0))
            else:
                w = -0.5 - 0.3 * (1.0 - front_min / (safe_dist + 1.0))
        
        # 侧方障碍物
        if left_min < safe_dist:
            w = min(w, -0.3)
        if right_min < safe_dist:
            w = max(w, 0.3)
        
        return np.array([v, w])


# =============================================================================
# 简单反应式策略（Simple Reactive）
# =============================================================================
@dataclass
class ReactiveConfig:
    """反应式策略配置"""
    v_max: float = 1.0
    w_max: float = 1.0
    safe_distance: float = 0.8
    robot_radius: float = 0.25


class ReactivePolicy:
    """
    简单反应式策略
    
    规则：前方有障碍就转，没有就走
    """
    
    def __init__(self, cfg: ReactiveConfig):
        self.cfg = cfg
        
    def reset(self):
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        lidar_m = obs[:180] * 10.0
        goal_angle = obs[183] * math.pi
        
        # 分区域获取最小距离
        left_min = np.min(lidar_m[0:60])
        front_min = np.min(lidar_m[60:120])
        right_min = np.min(lidar_m[120:180])
        
        safe_dist = self.cfg.safe_distance
        
        # 决策逻辑
        if front_min > safe_dist:
            # 前方安全，前进并调整朝向
            v = self.cfg.v_max
            w = np.clip(-goal_angle * 0.5, -self.cfg.w_max, self.cfg.w_max)
        else:
            # 前方有障碍，转向
            v = 0.0
            if left_min > right_min:
                w = 0.7  # 左转
            else:
                w = -0.7  # 右转
        
        return np.array([v, w], dtype=np.float32)


# =============================================================================
# 停滞检测配置
# =============================================================================
@dataclass
class StagnationConfig:
    """停滞检测配置"""
    velocity_threshold: float = 0.05    # 速度阈值：小于此值认为停滞
    angular_threshold: float = 0.3      # 角速度阈值：只有旋转没有前进也算停滞
    min_stagnation_steps: int = 3       # 最小停滞步数（避免噪声）


def detect_stagnation(velocity: float, angular: float, cfg: StagnationConfig) -> bool:
    """
    检测当前帧是否处于停滞状态
    
    停滞定义：线速度很低，或者只有旋转没有前进
    """
    return velocity < cfg.velocity_threshold or (
        velocity < cfg.velocity_threshold * 2 and abs(angular) > cfg.angular_threshold
    )


# =============================================================================
# 统一评估框架
# =============================================================================
def run_episode(env: UnityNavEnv, policy, policy_name: str = "Policy",
                stagnation_cfg: StagnationConfig = None) -> Dict:
    """运行单个 episode"""
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
    action_list = []
    stagnation_list = []  # 记录每帧是否停滞

    while not done:
        action = policy.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        ep_ret += reward
        ep_len += 1
        last_info = info
        
        min_lidar_list.append(float(np.min(obs[:180]) * 10.0))
        velocity_list.append(float(obs[184]))
        action_list.append(action.copy())
        
        # 检测停滞
        is_stagnant = detect_stagnation(
            float(obs[184]),  # 当前线速度
            float(action[1]),  # 当前角速度
            stagnation_cfg
        )
        stagnation_list.append(is_stagnant)

    # 计算停滞率
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
    """评估单个策略"""
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

    # 汇总统计
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
    """
    对比所有基线算法
    
    Args:
        env_cfg: 环境配置
        n_episodes: 每个策略评估的回合数
        save_dir: 结果保存目录
        
    Returns:
        comparison: 对比结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义所有策略
    policies = {
        "DWA": (DWAPolicy(DWAConfig()), DWAConfig()),
        "ORCA": (ORCAPolicy(ORCAConfig()), ORCAConfig()),
        "Reactive": (ReactivePolicy(ReactiveConfig()), ReactiveConfig()),
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
    print("BASELINE COMPARISON SUMMARY")
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
    
    # JSON 格式
    json_path = os.path.join(save_dir, f"baseline_comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # CSV 格式
    csv_path = os.path.join(save_dir, f"baseline_comparison_{timestamp}.csv")
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
    # 环境配置 - 使用相对路径
    import sys
    import os
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Windows 下的 Unity Build 路径
    build_path = os.path.join(script_dir, "Builds", "Project_1.exe")
    
    env_cfg = EnvConfig(
        file_name=build_path,
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=350,
        # Reward 参数（与 exp4 保持一致）
        progress_gain=2.5,
        time_penalty=-0.005,
        collision_penalty=-8.0,
        success_bonus=80.0,
        timeout_penalty=-15.0,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.15,
        action_l2_penalty=-0.0005,
    )
    
    # 运行对比评估
    results = compare_baselines(env_cfg, n_episodes=50, save_dir="./results")

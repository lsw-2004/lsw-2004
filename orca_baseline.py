"""
ORCA (Optimal Reciprocal Collision Avoidance) 基线算法

ORCA 是一种多智能体避障算法，通过构建速度障碍锥来选择安全速度。
在本实现中，我们使用简化的 ORCA 思想：基于 LiDAR 检测障碍物并选择避障速度。

注：完整的 ORCA 需要知道其他智能体的位置和速度信息，
     这里我们基于 LiDAR 观测实现一个简化版本。
"""
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from unity_env import UnityNavEnv, EnvConfig


@dataclass
class ORCAConfig:
    """ORCA 算法配置"""
    
    # 机器人参数
    robot_radius: float = 0.25          # 机器人半径
    v_max: float = 1.0                  # 最大线速度
    w_max: float = 1.0                  # 最大角速度
    
    # ORCA 参数
    time_horizon: float = 2.0           # 时间视野（秒）
    safety_margin: float = 0.15         # 安全余量
    
    # 速度采样
    n_v_samples: int = 11               # 线速度采样数
    n_w_samples: int = 21               # 角速度采样数
    
    # 目标吸引力
    goal_attraction: float = 1.5        # 目标吸引力权重
    
    # 速度障碍阈值
    obstacle_threshold: float = 0.3     # 认为是障碍物的 LiDAR 距离阈值（归一化）


class ORCAPolicy:
    """
    简化版 ORCA 策略
    
    基于 LiDAR 观测实现速度障碍避障：
    1. 根据 LiDAR 检测到的障碍物构建速度障碍
    2. 在安全速度空间中选择最接近目标方向的速度
    """
    
    def __init__(self, cfg: ORCAConfig):
        self.cfg = cfg
        self.prev_action = np.zeros(2, dtype=np.float32)
        
    def reset(self):
        """重置策略状态"""
        self.prev_action[:] = 0.0
        
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        根据观测选择动作
        
        Args:
            obs: 187 维观测向量
            
        Returns:
            action: 2 维动作向量 [线速度, 角速度]
        """
        # 解析观测
        lidar_norm = obs[:180]
        goal_dir = obs[180:182]
        goal_dist = obs[182] * 30.0
        goal_angle = obs[183] * math.pi
        cur_v = float(obs[184])
        cur_w = float(obs[185])
        
        # LiDAR 转米
        lidar_m = lidar_norm * 10.0
        
        # 检测障碍物
        obstacles = self._detect_obstacles(lidar_m)
        
        # 计算目标速度
        goal_speed = min(self.cfg.v_max, goal_dist / self.cfg.time_horizon)
        goal_vel = np.array([
            goal_speed * math.cos(goal_angle),
            goal_speed * math.sin(goal_angle)
        ])
        
        # 采样并选择最优安全速度
        best_action = self._select_best_velocity(
            obstacles, goal_vel, goal_angle, cur_v, cur_w
        )
        
        self.prev_action = best_action.copy()
        return best_action
    
    def _detect_obstacles(self, lidar_m: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        从 LiDAR 读数中检测障碍物
        
        Args:
            lidar_m: LiDAR 读数（米）
            
        Returns:
            obstacles: 障碍物列表 [(angle, distance, radius), ...]
        """
        obstacles = []
        
        # 将 LiDAR 分成多个扇区，每个扇区检测最近的障碍物
        n_sectors = 18
        sector_size = 180 // n_sectors
        
        for i in range(n_sectors):
            start_idx = i * sector_size
            end_idx = start_idx + sector_size
            sector = lidar_m[start_idx:end_idx]
            
            min_dist = float(np.min(sector))
            min_idx = int(np.argmin(sector))
            
            if min_dist < self.cfg.obstacle_threshold * 10.0:  # 阈值转换
                # 计算障碍物角度（相对于机器人前方）
                angle = math.radians((start_idx + min_idx) - 90)
                # 假设障碍物半径
                obstacle_radius = 0.2 if min_dist < 1.0 else 0.15
                obstacles.append((angle, min_dist, obstacle_radius))
        
        return obstacles
    
    def _select_best_velocity(
        self,
        obstacles: List[Tuple[float, float, float]],
        goal_vel: np.ndarray,
        goal_angle: float,
        cur_v: float,
        cur_w: float
    ) -> np.ndarray:
        """
        在安全速度空间中选择最优速度
        
        Args:
            obstacles: 障碍物列表
            goal_vel: 目标速度向量
            goal_angle: 目标角度
            cur_v: 当前线速度
            cur_w: 当前角速度
            
        Returns:
            best_action: 最优动作 [v, w]
        """
        best_score = -1e9
        best_action = np.array([0.0, 0.0], dtype=np.float32)
        
        # 采样线速度和角速度
        v_samples = np.linspace(0, self.cfg.v_max, self.cfg.n_v_samples)
        w_samples = np.linspace(-self.cfg.w_max, self.cfg.w_max, self.cfg.n_w_samples)
        
        for v in v_samples:
            for w in w_samples:
                # 检查是否安全
                if not self._is_velocity_safe(v, w, obstacles):
                    continue
                
                # 计算得分
                score = self._score_velocity(v, w, goal_vel, goal_angle)
                
                # 平滑性奖励
                smoothness = -0.1 * (abs(v - cur_v) + abs(w - cur_w))
                score += smoothness
                
                if score > best_score:
                    best_score = score
                    best_action[:] = [v, w]
        
        # 如果没有找到安全速度，减速或原地旋转
        if best_score == -1e9:
            # 选择角速度朝向目标方向，线速度为 0
            best_action[:] = [0.0, np.sign(goal_angle) * min(0.3, abs(goal_angle))]
            
        return best_action
    
    def _is_velocity_safe(self, v: float, w: float, obstacles: List[Tuple[float, float, float]]) -> bool:
        """
        检查给定速度是否安全（是否会碰撞）
        
        Args:
            v: 线速度
            w: 角速度
            obstacles: 障碍物列表
            
        Returns:
            is_safe: 是否安全
        """
        if len(obstacles) == 0:
            return True
            
        # 预测未来轨迹
        dt = 0.1
        n_steps = int(self.cfg.time_horizon / dt)
        
        for t in range(n_steps):
            # 预测位置（简化：假设直线运动 + 旋转）
            future_angle = w * t * dt
            future_dist = v * t * dt
            
            # 检查与每个障碍物的距离
            for obs_angle, obs_dist, obs_radius in obstacles:
                # 粗略估计：检查是否会在障碍物附近经过
                angle_diff = abs(future_angle - obs_angle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                
                # 如果角度接近且距离会增加碰撞风险
                if angle_diff < 0.5:  # 约 30 度
                    safe_dist = (self.cfg.robot_radius + obs_radius + 
                                self.cfg.safety_margin + future_dist * 0.3)
                    if obs_dist < safe_dist:
                        return False
        
        return True
    
    def _score_velocity(self, v: float, w: float, goal_vel: np.ndarray, goal_angle: float) -> float:
        """
        评估速度的得分
        
        Args:
            v: 线速度
            w: 角速度
            goal_vel: 目标速度向量
            goal_angle: 目标角度
            
        Returns:
            score: 得分
        """
        # 目标接近度：速度方向与目标方向的一致性
        vel_angle = w * 0.5  # 粗略估计速度方向
        heading_score = 1.0 - abs(goal_angle - vel_angle) / math.pi
        
        # 速度大小奖励
        speed_score = v / self.cfg.v_max
        
        # 角度接近奖励
        angle_score = 1.0 - min(abs(goal_angle), abs(w - goal_angle)) / self.cfg.w_max
        
        # 综合得分
        score = (
            self.cfg.goal_attraction * heading_score +
            0.5 * speed_score +
            0.3 * angle_score
        )
        
        return score


def run_orca_episode(env: UnityNavEnv, policy: ORCAPolicy) -> Dict:
    """
    运行一个 ORCA episode
    
    Args:
        env: Unity 环境
        policy: ORCA 策略
        
    Returns:
        result: 包含各种指标的字典
    """
    obs, info = env.reset()
    policy.reset()

    done = False
    ep_ret = 0.0
    ep_len = 0
    last_info = {}
    
    min_lidar_list = []
    velocity_list = []

    while not done:
        action = policy.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        ep_ret += reward
        ep_len += 1
        last_info = info
        
        min_lidar_list.append(float(np.min(obs[:180]) * 10.0))
        velocity_list.append(float(obs[184]))

    result = {
        "return": ep_ret,
        "length": ep_len,
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "timeout": bool(last_info.get("timeout", False)),
        "final_goal_dist": float(last_info.get("goal_dist", np.nan)),
        "min_lidar_mean": float(np.mean(min_lidar_list)) if min_lidar_list else 0.0,
        "velocity_mean": float(np.mean(velocity_list)) if velocity_list else 0.0,
    }
    return result


def evaluate_orca(env_cfg: EnvConfig, policy_cfg: ORCAConfig, n_episodes: int = 50) -> Dict:
    """
    评估 ORCA 策略
    
    Args:
        env_cfg: 环境配置
        policy_cfg: ORCA 配置
        n_episodes: 评估回合数
        
    Returns:
        summary: 评估结果汇总
    """
    env = UnityNavEnv(env_cfg)
    policy = ORCAPolicy(policy_cfg)

    results = []
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"ORCA Baseline Evaluation")
    print(f"{'='*50}")
    print(f"Episodes: {n_episodes}")
    print(f"Environment: {env_cfg.file_name}")
    print(f"{'='*50}\n")
    
    for i in range(n_episodes):
        result = run_orca_episode(env, policy)
        results.append(result)
        
        status = "SUCCESS" if result["success"] else ("COLLISION" if result["collision"] else "TIMEOUT")
        print(f"[ORCA][{i+1:03d}] len={result['length']:3d} | "
              f"dist={result['final_goal_dist']:.2f}m | {status}")

    env.close()
    eval_time = time.time() - start_time

    # 汇总统计
    success_rate = np.mean([r["success"] for r in results])
    collision_rate = np.mean([r["collision"] for r in results])
    timeout_rate = np.mean([r["timeout"] for r in results])
    avg_len = np.mean([r["length"] for r in results])
    avg_ret = np.mean([r["return"] for r in results])
    avg_goal_dist = np.nanmean([r["final_goal_dist"] for r in results])
    avg_min_lidar = np.mean([r["min_lidar_mean"] for r in results])
    avg_velocity = np.mean([r["velocity_mean"] for r in results])
    
    std_len = np.std([r["length"] for r in results])
    std_ret = np.std([r["return"] for r in results])

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
        "avg_velocity": avg_velocity,
        "n_episodes": n_episodes,
        "eval_time": eval_time,
    }

    print(f"\n{'='*50}")
    print("ORCA Evaluation Summary")
    print(f"{'='*50}")
    print(f"success_rate      : {success_rate:.3f}")
    print(f"collision_rate    : {collision_rate:.3f}")
    print(f"timeout_rate      : {timeout_rate:.3f}")
    print(f"avg_length        : {avg_len:.1f} ± {std_len:.1f}")
    print(f"avg_return        : {avg_ret:.3f} ± {std_ret:.3f}")
    print(f"avg_final_dist    : {avg_goal_dist:.2f} m")
    print(f"avg_min_lidar     : {avg_min_lidar:.2f} m")
    print(f"avg_velocity      : {avg_velocity:.3f}")
    print(f"eval_time         : {eval_time:.1f} s")
    print(f"{'='*50}\n")

    return summary


if __name__ == "__main__":
    # 环境配置（与 PPO 训练保持一致）
    env_cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\Builds\Project_1.exe",  # 修改为你的路径
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=350,
        # Reward 参数（与 PPO 保持一致）
        progress_gain=2.5,
        time_penalty=-0.005,
        collision_penalty=-8.0,
        success_bonus=80.0,
        timeout_penalty=-15.0,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.15,
        action_l2_penalty=-0.0005,
    )

    policy_cfg = ORCAConfig()
    
    # 运行评估
    summary = evaluate_orca(env_cfg, policy_cfg, n_episodes=50)

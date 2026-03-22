"""
DWA (Dynamic Window Approach) 基线算法

用于与 PPO 强化学习方法进行对比实验。
DWA 是一种基于规则的局部避障算法，通过采样速度空间并评分来选择最优动作。
"""
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from unity_env import UnityNavEnv, EnvConfig


@dataclass
class DWAConfig:
    """DWA 算法配置"""
    
    # 机器人控制上限（与 Unity 中 RobotController 的速度上限一致）
    v_max: float = 1.0          # 最大线速度
    w_max: float = 1.0          # 最大角速度

    # 采样参数
    n_v: int = 9                # 线速度采样数
    n_w: int = 15               # 角速度采样数

    # 预测时域
    horizon_steps: int = 10     # 预测步数
    dt: float = 0.1             # 时间步长

    # 评分权重
    heading_weight: float = 1.5     # 朝向目标权重
    clearance_weight: float = 2.0   # 避障安全权重
    velocity_weight: float = 0.3    # 速度权重
    goal_weight: float = 1.2        # 目标距离权重

    # 安全参数
    robot_radius: float = 0.25      # 机器人半径
    brake_margin: float = 0.08      # 刹车安全余量
    hard_clearance: float = 0.20    # 硬性安全距离阈值

    # 动作平滑（限制相邻帧动作变化）
    max_delta_v: float = 0.3        # 线速度最大变化
    max_delta_w: float = 0.4        # 角速度最大变化


class DWAPolicy:
    """
    DWA 策略实现
    
    基于当前 187 维观测直接做 DWA 决策。
    
    观测结构:
      0:180    lidar (已归一化到 [0,1]，最大距离 10m)
      180      goal_dir_x (目标方向 x 分量)
      181      goal_dir_z (目标方向 z 分量)
      182      goal_dist_norm (归一化目标距离，_maxScenarioSize=30)
      183      goal_angle_norm (-1~1 对应 -π~π)
      184      linear_vel_norm (归一化线速度)
      185      angular_vel_norm (归一化角速度)
      186      collision_flag (碰撞标志)
    """
    
    def __init__(self, cfg: DWAConfig):
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
        goal_dist = obs[182] * 30.0      # 还原为米
        goal_angle = obs[183] * math.pi  # 还原为弧度
        cur_v = float(obs[184])          # 当前线速度（归一化）
        cur_w = float(obs[185])          # 当前角速度（归一化）
        
        # LiDAR 归一化值转米（Unity 中 maxDistance = 10m）
        lidar_m = lidar_norm * 10.0

        # 生成候选速度（在动态窗口内采样）
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

    def _score_action(
        self,
        v: float,
        w: float,
        goal_angle: float,
        goal_dist: float,
        lidar_m: np.ndarray
    ) -> float:
        """
        评估候选动作的得分
        
        Args:
            v: 候选线速度
            w: 候选角速度
            goal_angle: 目标角度
            goal_dist: 目标距离
            lidar_m: LiDAR 读数（米）
            
        Returns:
            score: 综合得分
        """
        # 1) heading score: 预测末端朝向与目标夹角越小越好
        pred_heading = w * self.cfg.horizon_steps * self.cfg.dt
        heading_err = abs(self._wrap_to_pi(goal_angle - pred_heading))
        heading_score = 1.0 - min(heading_err / math.pi, 1.0)

        # 2) clearance score: 前方障碍物余量
        clearance = self._estimate_clearance(v, w, lidar_m)
        if clearance < self.cfg.hard_clearance:
            return -1e6  # 不安全，直接拒绝
        
        clearance_score = np.tanh(clearance / 2.0)  # 归一化

        # 3) velocity score: 鼓励前进
        velocity_score = max(v, 0.0) / self.cfg.v_max

        # 4) goal score: 根据距离调整权重
        if goal_dist < 3.0:
            # 近目标时，强调朝向
            goal_score = 0.5 + 0.5 * heading_score
        else:
            # 远目标时，鼓励前进
            goal_score = np.tanh(goal_dist / 10.0)

        # 综合得分
        score = (
            self.cfg.heading_weight * heading_score +
            self.cfg.clearance_weight * clearance_score +
            self.cfg.velocity_weight * velocity_score +
            self.cfg.goal_weight * goal_score
        )

        # 轻微惩罚大角速度，提高平稳性
        score -= 0.03 * abs(w)
        
        # 前进时给予额外奖励
        if v > 0.1:
            score += 0.1
            
        return float(score)

    def _estimate_clearance(self, v: float, w: float, lidar_m: np.ndarray) -> float:
        """
        估计当前速度方向上的安全余量
        
        Args:
            v: 线速度
            w: 角速度
            lidar_m: LiDAR 读数（米）
            
        Returns:
            clearance: 安全余量（米）
        """
        # LiDAR 中间位置对应正前方（索引 90）
        center_deg = 90
        
        # 根据转向趋势调整关注扇区
        steer_offset = int(np.clip(w, -1.0, 1.0) * 30)
        center = center_deg + steer_offset

        # 扇区宽度：转弯时更宽
        half_width = 15 if abs(w) < 0.3 else 25
        left = max(0, center - half_width)
        right = min(179, center + half_width)

        sector = lidar_m[left:right + 1]
        if len(sector) == 0:
            return 0.0

        # 最小障碍物距离
        raw_clearance = float(np.min(sector))
        
        # 减去机器人半径和刹车余量
        clearance = raw_clearance - self.cfg.robot_radius - self.cfg.brake_margin

        # 速度越大，需要更多安全余量
        if v > 0:
            clearance -= v * 0.2
            
        return clearance

    @staticmethod
    def _wrap_to_pi(x: float) -> float:
        """将角度归一化到 [-π, π]"""
        return (x + math.pi) % (2 * math.pi) - math.pi


def run_dwa_episode(env: UnityNavEnv, policy: DWAPolicy) -> Dict:
    """
    运行一个 DWA episode
    
    Args:
        env: Unity 环境
        policy: DWA 策略
        
    Returns:
        result: 包含各种指标的字典
    """
    obs, info = env.reset()
    policy.reset()

    done = False
    ep_ret = 0.0
    ep_len = 0
    last_info = {}
    
    # 记录更详细的轨迹信息
    min_lidar_list = []
    velocity_list = []

    while not done:
        action = policy.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        ep_ret += reward
        ep_len += 1
        last_info = info
        
        # 记录轨迹数据
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


def evaluate_dwa(env_cfg: EnvConfig, policy_cfg: DWAConfig, n_episodes: int = 50) -> Dict:
    """
    评估 DWA 策略
    
    Args:
        env_cfg: 环境配置
        policy_cfg: DWA 配置
        n_episodes: 评估回合数
        
    Returns:
        summary: 评估结果汇总
    """
    env = UnityNavEnv(env_cfg)
    policy = DWAPolicy(policy_cfg)

    results = []
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"DWA Baseline Evaluation")
    print(f"{'='*50}")
    print(f"Episodes: {n_episodes}")
    print(f"Environment: {env_cfg.file_name}")
    print(f"{'='*50}\n")
    
    for i in range(n_episodes):
        result = run_dwa_episode(env, policy)
        results.append(result)
        
        status = "SUCCESS" if result["success"] else ("COLLISION" if result["collision"] else "TIMEOUT")
        print(f"[DWA][{i+1:03d}] len={result['length']:3d} | "
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
    print("DWA Evaluation Summary")
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
        # Reward 参数（与 exp3/exp4 保持一致，便于公平对比）
        progress_gain=2.5,
        time_penalty=-0.005,
        collision_penalty=-8.0,
        success_bonus=80.0,
        timeout_penalty=-15.0,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.15,
        action_l2_penalty=-0.0005,
    )

    policy_cfg = DWAConfig()
    
    # 运行评估
    summary = evaluate_dwa(env_cfg, policy_cfg, n_episodes=50)

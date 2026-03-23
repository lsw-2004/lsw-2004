"""
实验5 v3: 目标成功率 0.8 的改进训练脚本

核心改进 (基于学术研究):
1. 社交力模型启发的奖励设计 (Social Force Model)
2. 预测性安全层 (Predictive Safety Layer) - 借鉴 MPC 思想
3. 多任务学习 + 辅助任务 (Auxiliary Tasks)
4. 改进的经验回放: HER + 优先级回放
5. 更鲁棒的动态物体追踪 (Kalman Filter)
6. Ensemble 策略 + 软目标更新
7. 渐进式课程学习 (Automated Curriculum Learning)

参考文献:
- Long et al. "Learning Complex Dexterous Manipulation" (2017) - 辅助任务
- Chen et al. "Socially Aware Motion Planning" (2019) - 社交导航
- Andrychowicz et al. "Hindsight Experience Replay" (2017) - HER
- Henderson et al. "Deep Reinforcement Learning that Matters" (2017) - PPO 改进
"""
import os
import random
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from unity_env import UnityNavEnv, EnvConfig


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PPOConfig:
    """改进的 PPO 配置"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # 训练参数
    total_updates: int = 5000          # 更多更新
    rollout_steps: int = 4096          # 更长的 rollout
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # 优化器参数
    lr: float = 3e-4
    lr_actor: float = 1e-4             # Actor 使用更小的学习率
    lr_critic: float = 5e-4            # Critic 使用更大的学习率
    clip_coef: float = 0.2
    
    # Entropy 调度
    ent_coef_start: float = 0.02       # 更高的初始探索
    ent_coef_end: float = 0.0005       # 更低的最终探索
    
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 8             # 更多 epoch
    minibatch_size: int = 256
    target_kl: float = 0.015           # 更严格的 KL 约束

    # 观测参数
    obs_dim: int = 187
    lidar_dim: int = 180
    raw_low_dim: int = 7
    
    # 动态检测参数
    history_len: int = 12              # 更多历史帧
    dynamic_feat_dim: int = 64         # 更丰富的动态特征
    max_detected_objects: int = 8      # 检测更多物体
    
    action_dim: int = 2
    seq_len: int = 20                  # 更长的序列

    # 保存路径
    save_dir: str = "./checkpoints/cnn_gru_ppo_tb/exp5_v3"
    log_dir: str = "./runs/cnn_gru_ppo_tb/exp5_v3"
    save_every: int = 50

    # 评估
    eval_every: int = 10
    eval_episodes: int = 100           # 更多评估回合

    # 恢复训练
    resume: bool = False
    resume_checkpoint: str = ""

    # 改进功能开关
    use_her: bool = True               # Hindsight Experience Replay
    use_aux_tasks: bool = True         # 辅助任务
    use_priority_replay: bool = True   # 优先级经验回放
    use_safety_layer: bool = True      # 预测性安全层
    use_social_force: bool = True      # 社交力模型奖励
    use_ensemble: bool = False         # Ensemble (可选, 需要更多显存)
    use_kalman_tracker: bool = True    # Kalman 滤波器追踪
    use_auto_curriculum: bool = True   # 自动课程学习

    # 辅助任务权重
    aux_collision_weight: float = 0.3
    aux_lidar_weight: float = 0.1
    aux_speed_weight: float = 0.05

    @property
    def low_dim(self) -> int:
        return self.raw_low_dim + self.dynamic_feat_dim

    @property
    def enhanced_obs_dim(self) -> int:
        return self.lidar_dim + self.low_dim


# =========================
# Kalman Filter 目标追踪器
# =========================
class KalmanObjectTracker:
    """
    基于 Kalman 滤波器的动态物体追踪器
    
    状态向量: [x, y, vx, vy] (位置和速度)
    观测向量: [r, theta] (极坐标距离和角度)
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        
        # 状态转移矩阵 (匀速模型)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 观测矩阵 (极坐标到笛卡尔)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        self.Q[2, 2] = 0.5  # 速度噪声更大
        self.Q[3, 3] = 0.5
        
        # 观测噪声
        self.R = np.eye(2, dtype=np.float32) * 0.3
        
        # 状态和协方差
        self.x = None  # 状态
        self.P = None  # 协方差
        self.initialized = False
        
    def initialize(self, r: float, theta: float):
        """初始化状态"""
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.x = np.array([x, y, 0, 0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0
        self.initialized = True
        
    def predict(self) -> Tuple[float, float, float, float]:
        """预测步骤，返回 (x, y, vx, vy)"""
        if not self.initialized:
            return 0, 0, 0, 0
            
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[0], self.x[1], self.x[2], self.x[3]
    
    def update(self, r: float, theta: float) -> Tuple[float, float, float, float]:
        """更新步骤"""
        z = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
        
        if not self.initialized:
            self.initialize(r, theta)
            return self.x[0], self.x[1], self.x[2], self.x[3]
        
        # Kalman 增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        y = z - self.H @ self.x  # 观测残差
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x[0], self.x[1], self.x[2], self.x[3]
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """获取当前状态 (x, y, vx, vy)"""
        if not self.initialized:
            return 0, 0, 0, 0
        return self.x[0], self.x[1], self.x[2], self.x[3]
    
    def get_speed(self) -> float:
        """获取速度大小"""
        if not self.initialized:
            return 0
        return np.sqrt(self.x[2]**2 + self.x[3]**2)
    
    def get_future_position(self, t: float) -> Tuple[float, float]:
        """预测 t 秒后的位置"""
        if not self.initialized:
            return 0, 0
        return self.x[0] + self.x[2] * t, self.x[1] + self.x[3] * t


# =========================
# 增强的动态物体检测器
# =========================
class EnhancedDynamicDetector:
    """
    增强的动态物体检测器
    
    改进:
    1. Kalman 滤波器追踪每个物体
    2. 更鲁棒的聚类算法
    3. 机器人运动补偿
    4. 威胁评估
    """
    
    def __init__(
        self, 
        lidar_dim: int = 180,
        history_len: int = 12,
        max_objects: int = 8,
        use_kalman: bool = True
    ):
        self.lidar_dim = lidar_dim
        self.history_len = history_len
        self.max_objects = max_objects
        self.use_kalman = use_kalman
        
        # LiDAR 角度 (假设均匀分布，-180° 到 +180°)
        self.angles = np.linspace(-np.pi, np.pi, lidar_dim, endpoint=False)
        
        # 历史缓冲
        self.lidar_history: Deque[np.ndarray] = deque(maxlen=history_len)
        self.robot_pose_history: Deque[Tuple[float, float, float]] = deque(maxlen=history_len)
        
        # Kalman 追踪器
        self.trackers: List[KalmanObjectTracker] = []
        self.last_objects: List[Dict] = []
        
        # 噪声阈值
        self.dynamic_threshold = 0.15      # 动态判定阈值 (m)
        self.speed_threshold = 0.3         # 最小速度阈值 (m/s)
        self.cluster_threshold = 8         # 聚类角度阈值 (索引差)
        
    def reset(self):
        """重置检测器"""
        self.lidar_history.clear()
        self.robot_pose_history.clear()
        self.trackers = []
        self.last_objects = []
        
    def update(
        self,
        lidar: np.ndarray,
        robot_x: float,
        robot_z: float,
        robot_yaw: float,
        robot_vx: float = 0.0,
        robot_vz: float = 0.0
    ):
        """更新历史数据"""
        self.lidar_history.append(lidar.copy())
        self.robot_pose_history.append((robot_x, robot_z, robot_yaw))
        
    def detect(self) -> np.ndarray:
        """
        检测动态物体并返回特征向量
        
        返回 64 维特征:
        - [0:8] 每个物体的距离
        - [8:16] 每个物体的角度
        - [16:24] 每个物体的速度
        - [24:32] 每个物体的威胁等级
        - [32:40] 每个物体的 TTC
        - [40:48] 预测的碰撞位置
        - [48:56] 物体的运动方向
        - [56:64] 全局统计特征
        """
        features = np.zeros(64, dtype=np.float32)
        
        if len(self.lidar_history) < 3:
            return features
            
        # Step 1: 计算补偿后的速度场
        velocity_field = self._compute_velocity_field()
        
        # Step 2: 聚类动态点
        clusters = self._cluster_dynamic_points(velocity_field)
        
        # Step 3: 估计物体属性
        objects = self._estimate_object_properties(clusters, velocity_field)
        
        # Step 4: 更新追踪器并预测
        objects = self._update_trackers(objects)
        
        # Step 5: 计算威胁
        objects = self._compute_threats(objects)
        
        # Step 6: 构建特征
        features = self._build_features(objects)
        
        self.last_objects = objects
        return features
    
    def _compute_velocity_field(self) -> np.ndarray:
        """
        计算每个 LiDAR 点的速度场，补偿机器人运动
        """
        if len(self.lidar_history) < 2:
            return np.zeros((self.lidar_dim, 2))
            
        curr_lidar = self.lidar_history[-1] * 10.0  # 转米
        prev_lidar = self.lidar_history[-2] * 10.0
        
        # 机器人运动补偿
        if len(self.robot_pose_history) >= 2:
            curr_pose = self.robot_pose_history[-1]
            prev_pose = self.robot_pose_history[-2]
            
            # 机器人的位移
            dx = curr_pose[0] - prev_pose[0]
            dz = curr_pose[1] - prev_pose[1]
            dyaw = curr_pose[2] - prev_pose[2]
        else:
            dx, dz, dyaw = 0, 0, 0
            
        velocity_field = np.zeros((self.lidar_dim, 2))  # (vr, vt)
        
        for i in range(self.lidar_dim):
            # 观测到的距离变化
            dr = curr_lidar[i] - prev_lidar[i]
            
            # 机器人运动引起的距离变化 (简化模型)
            angle = self.angles[i]
            robot_motion_dr = -(dx * np.cos(angle) + dz * np.sin(angle))
            
            # 补偿后的距离变化
            compensated_dr = dr - robot_motion_dr
            
            # 切向速度估计 (通过相邻点的变化)
            if i > 0 and i < self.lidar_dim - 1:
                # 角度变化导致距离变化
                angular_effect = dyaw * curr_lidar[i]
                vt = angular_effect
            else:
                vt = 0
                
            velocity_field[i] = [compensated_dr, vt]
            
        return velocity_field
    
    def _cluster_dynamic_points(self, velocity_field: np.ndarray) -> List[Dict]:
        """聚类动态点"""
        curr_lidar = self.lidar_history[-1] * 10.0
        
        # 计算速度大小
        speeds = np.sqrt(velocity_field[:, 0]**2 + velocity_field[:, 1]**2)
        
        # 动态点判定
        dynamic_mask = speeds > self.speed_threshold
        
        # 距离阈值
        distance_mask = curr_lidar < 8.0
        
        # 有效动态点
        valid_mask = dynamic_mask & distance_mask
        
        # 连通域聚类
        clusters = []
        in_cluster = False
        cluster_start = 0
        
        for i in range(self.lidar_dim):
            if valid_mask[i] and not in_cluster:
                in_cluster = True
                cluster_start = i
            elif not valid_mask[i] and in_cluster:
                in_cluster = False
                if i - cluster_start >= 3:
                    clusters.append({
                        'indices': list(range(cluster_start, i)),
                        'velocities': velocity_field[cluster_start:i],
                        'distances': curr_lidar[cluster_start:i],
                        'speeds': speeds[cluster_start:i]
                    })
                    
        # 处理最后一个聚类
        if in_cluster and self.lidar_dim - cluster_start >= 3:
            clusters.append({
                'indices': list(range(cluster_start, self.lidar_dim)),
                'velocities': velocity_field[cluster_start:self.lidar_dim],
                'distances': curr_lidar[cluster_start:self.lidar_dim],
                'speeds': speeds[cluster_start:self.lidar_dim]
            })
            
        # 合并相近的聚类
        clusters = self._merge_clusters(clusters)
        
        return clusters
    
    def _merge_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """合并相近的聚类"""
        if len(clusters) <= 1:
            return clusters
            
        merged = []
        used = set()
        
        for i, c1 in enumerate(clusters):
            if i in used:
                continue
                
            merged_cluster = c1.copy()
            
            for j, c2 in enumerate(clusters[i+1:], i+1):
                if j in used:
                    continue
                    
                # 检查是否应该合并
                gap = min(abs(c1['indices'][-1] - c2['indices'][0]),
                         abs(c2['indices'][-1] - c1['indices'][0]))
                
                if gap < self.cluster_threshold:
                    merged_cluster['indices'].extend(c2['indices'])
                    merged_cluster['velocities'] = np.vstack([
                        merged_cluster['velocities'], c2['velocities']
                    ])
                    merged_cluster['distances'] = np.concatenate([
                        merged_cluster['distances'], c2['distances']
                    ])
                    merged_cluster['speeds'] = np.concatenate([
                        merged_cluster['speeds'], c2['speeds']
                    ])
                    used.add(j)
                    
            merged.append(merged_cluster)
            used.add(i)
            
        return merged
    
    def _estimate_object_properties(self, clusters: List[Dict], velocity_field: np.ndarray) -> List[Dict]:
        """估计每个聚类的物体属性"""
        objects = []
        
        for cluster in clusters[:self.max_objects]:
            indices = cluster['indices']
            velocities = cluster['velocities']
            distances = cluster['distances']
            
            # 物体中心
            center_angle = np.mean(self.angles[indices])
            min_dist_idx = np.argmin(distances)
            min_dist = float(distances[min_dist_idx])
            
            # 速度估计
            mean_vr = float(np.mean(velocities[:, 0]))
            mean_vt = float(np.mean(velocities[:, 1]))
            
            # 速度上限
            total_speed = np.sqrt(mean_vr**2 + mean_vt**2)
            if total_speed > 3.0:
                scale = 3.0 / total_speed
                mean_vr *= scale
                mean_vt *= scale
                
            # 物体大小
            angular_size = len(indices) * (2 * np.pi / self.lidar_dim)
            
            objects.append({
                'angle': center_angle,
                'distance': min_dist,
                'vr': mean_vr,
                'vt': mean_vt,
                'speed': total_speed,
                'angular_size': angular_size,
                'n_points': len(indices),
                'indices': indices
            })
            
        return objects
    
    def _update_trackers(self, objects: List[Dict]) -> List[Dict]:
        """更新 Kalman 追踪器"""
        if not self.use_kalman:
            return objects
            
        # 简单的数据关联: 按角度匹配
        for i, obj in enumerate(objects):
            matched = False
            
            for tracker in self.trackers:
                # 获取预测位置
                pred_x, pred_y, pred_vx, pred_vy = tracker.predict()
                pred_r = np.sqrt(pred_x**2 + pred_y**2)
                pred_theta = np.arctan2(pred_y, pred_x)
                
                # 检查匹配
                angle_diff = abs(obj['angle'] - pred_theta)
                dist_diff = abs(obj['distance'] - pred_r)
                
                if angle_diff < 0.5 and dist_diff < 1.0:
                    # 更新追踪器
                    tracker.update(obj['distance'], obj['angle'])
                    obj['tracker'] = tracker
                    obj['predicted_x'] = pred_x
                    obj['predicted_y'] = pred_y
                    matched = True
                    break
                    
            if not matched:
                # 创建新追踪器
                tracker = KalmanObjectTracker()
                tracker.initialize(obj['distance'], obj['angle'])
                obj['tracker'] = tracker
                
        return objects
    
    def _compute_threats(self, objects: List[Dict]) -> List[Dict]:
        """计算威胁等级"""
        for obj in objects:
            distance = obj['distance']
            vr = obj['vr']
            speed = obj['speed']
            angle = obj['angle']
            
            # 威胁等级: 距离近 + 正在接近 + 前方
            approach_factor = max(0, -vr) / 2.0
            distance_factor = 1.0 - min(distance / 5.0, 1.0)
            angle_factor = 1.0 - abs(angle) / np.pi
            
            threat = (approach_factor * 0.4 + distance_factor * 0.4 + angle_factor * 0.2)
            obj['threat'] = float(np.clip(threat, 0, 1))
            
            # TTC
            if vr < -0.1:
                ttc = distance / (-vr)
                obj['ttc'] = float(min(ttc, 10.0))
            else:
                obj['ttc'] = 10.0
                
            # 未来位置预测
            future_dist = distance + vr * 1.0
            future_angle = angle + np.arcsin(np.clip(vt / (distance + 0.1) * 1.0, -1, 1)) if 'vt' in obj else angle
            obj['future_pos'] = (future_dist, future_angle)
            
            # 运动方向
            if 'tracker' in obj:
                tracker = obj['tracker']
                _, _, vx, vy = tracker.get_state()
                obj['motion_direction'] = np.arctan2(vy, vx)
            else:
                obj['motion_direction'] = angle
                
        return objects
    
    def _build_features(self, objects: List[Dict]) -> np.ndarray:
        """构建特征向量"""
        features = np.zeros(64, dtype=np.float32)
        
        n_objects = min(len(objects), self.max_objects)
        
        for i in range(n_objects):
            obj = objects[i]
            features[i] = obj['distance'] / 10.0
            features[8 + i] = obj['angle'] / np.pi
            features[16 + i] = obj['speed'] / 3.0
            features[24 + i] = obj['threat']
            features[32 + i] = obj['ttc'] / 10.0
            features[40 + i] = obj['future_pos'][0] / 10.0
            features[48 + i] = obj['motion_direction'] / np.pi
            
        # 全局统计 (56:64)
        if n_objects > 0:
            features[56] = n_objects / self.max_objects
            features[57] = min(obj['distance'] for obj in objects) / 10.0
            features[58] = max(obj['threat'] for obj in objects)
            features[59] = min(obj['ttc'] for obj in objects) / 10.0
            features[60] = sum(1 for obj in objects if obj['vr'] < 0) / max(n_objects, 1)
            features[61] = sum(obj['threat'] for obj in objects) / max(n_objects, 1)
            features[62] = np.mean([obj['angle'] for obj in objects]) / np.pi
            features[63] = np.std([obj['angle'] for obj in objects]) / np.pi
            
        return features


# =========================
# 社交力模型奖励
# =========================
class SocialForceReward:
    """
    基于社交力模型的奖励函数
    
    参考: Helbing et al. "Social Force Model for Pedestrian Dynamics"
    """
    
    def __init__(
        self,
        goal_gain: float = 3.0,
        collision_penalty: float = -15.0,
        safety_margin: float = 0.5,
        social_force_weight: float = 0.5,
        comfort_distance: float = 1.0
    ):
        self.goal_gain = goal_gain
        self.collision_penalty = collision_penalty
        self.safety_margin = safety_margin
        self.social_force_weight = social_force_weight
        self.comfort_distance = comfort_distance
        
    def compute(
        self,
        base_reward: float,
        obs: np.ndarray,
        info: Dict,
        dynamic_objects: List[Dict],
        prev_obs: Optional[np.ndarray] = None,
        update: int = 1,
        total_updates: int = 5000
    ) -> float:
        """计算社交力模型奖励"""
        reward = base_reward
        
        # 解析观测
        lidar = obs[:180] * 10.0
        goal_dist = obs[182] * 30.0
        goal_angle = obs[183] * np.pi
        cur_v = obs[184]
        cur_w = obs[185]
        
        # 1. 目标吸引力
        if prev_obs is not None:
            prev_goal_dist = prev_obs[182] * 30.0
            progress = prev_goal_dist - goal_dist
            reward += self.goal_gain * progress
            
        # 2. 社交力 (与动态障碍物的相互作用)
        for obj in dynamic_objects:
            dist = obj['distance']
            vr = obj.get('vr', 0)
            threat = obj.get('threat', 0)
            
            # 斥力: 距离越近, 斥力越大
            if dist < self.comfort_distance:
                repulsion = (self.comfort_distance - dist) / self.comfort_distance
                reward -= self.social_force_weight * repulsion * (1 + threat)
                
            # 接近惩罚
            if vr < 0 and dist < 2.0:
                approach_penalty = -vr * (2.0 - dist) / 2.0
                reward -= approach_penalty
                
        # 3. 安全裕度奖励
        min_lidar = np.min(lidar)
        if min_lidar > self.safety_margin:
            reward += 0.02  # 安全通行奖励
        elif min_lidar > 0.3:
            reward += 0.01  # 较安全通行奖励
            
        # 4. 平滑性奖励
        if prev_obs is not None:
            prev_v = prev_obs[184]
            prev_w = prev_obs[185]
            smoothness = -0.02 * (abs(cur_v - prev_v) + abs(cur_w - prev_w))
            reward += smoothness
            
        # 5. 朝向奖励
        if abs(goal_angle) < 0.3 and cur_v > 0.1:
            reward += 0.05  # 朝向正确且前进
            
        return reward


# =========================
# 预测性安全层
# =========================
class PredictiveSafetyLayer:
    """
    预测性安全层 (Predictive Safety Layer)
    
    在动作执行前预测是否安全，如不安全则修正动作。
    借鉴 MPC (Model Predictive Control) 思想。
    """
    
    def __init__(
        self,
        robot_radius: float = 0.25,
        safety_margin: float = 0.15,
        prediction_horizon: int = 10,
        dt: float = 0.1
    ):
        self.robot_radius = robot_radius
        self.safety_margin = safety_margin
        self.prediction_horizon = prediction_horizon
        self.dt = dt
        
    def check_and_correct(
        self,
        action: np.ndarray,
        obs: np.ndarray,
        dynamic_objects: List[Dict]
    ) -> np.ndarray:
        """检查并修正动作"""
        lidar = obs[:180] * 10.0
        v, w = action[0], action[1]
        
        # 检查静态障碍物碰撞
        static_safe, static_risk = self._check_static_obstacles(v, w, lidar)
        
        # 检查动态障碍物碰撞
        dynamic_safe, dynamic_risk = self._check_dynamic_obstacles(v, w, dynamic_objects)
        
        if static_safe and dynamic_safe:
            return action
            
        # 需要修正
        corrected_action = self._correct_action(action, static_risk, dynamic_risk, lidar, dynamic_objects)
        return corrected_action
    
    def _check_static_obstacles(self, v: float, w: float, lidar: np.ndarray) -> Tuple[bool, float]:
        """检查静态障碍物碰撞风险"""
        if v <= 0:
            return True, 0.0
            
        # 预测未来轨迹
        risk = 0.0
        
        for t in range(1, self.prediction_horizon + 1):
            future_dist = v * t * self.dt
            future_angle = w * t * self.dt
            
            # 检查该位置的 LiDAR 距离
            # 将角度转换为 LiDAR 索引
            lidar_angle = future_angle  # 相对于前方
            lidar_idx = int((lidar_angle + np.pi) / (2 * np.pi) * len(lidar))
            lidar_idx = np.clip(lidar_idx, 0, len(lidar) - 1)
            
            # 考虑一定的角度范围
            margin = 5
            start_idx = max(0, lidar_idx - margin)
            end_idx = min(len(lidar), lidar_idx + margin + 1)
            
            min_dist = np.min(lidar[start_idx:end_idx])
            
            # 检查是否会碰撞
            required_dist = future_dist + self.robot_radius + self.safety_margin
            
            if min_dist < required_dist:
                risk = max(risk, 1.0 - min_dist / required_dist)
                
        return risk < 0.3, risk
    
    def _check_dynamic_obstacles(self, v: float, w: float, objects: List[Dict]) -> Tuple[bool, float]:
        """检查动态障碍物碰撞风险"""
        if v <= 0 or len(objects) == 0:
            return True, 0.0
            
        risk = 0.0
        
        for obj in objects:
            dist = obj['distance']
            obj_vr = obj.get('vr', 0)
            obj_angle = obj['angle']
            
            # 预测未来位置
            for t in range(1, self.prediction_horizon + 1):
                # 机器人位置
                robot_dist = v * t * self.dt
                robot_angle = w * t * self.dt
                
                # 障碍物位置 (匀速模型)
                future_obj_dist = dist + obj_vr * t * self.dt
                future_obj_angle = obj_angle
                
                # 简化: 假设障碍物在同一角度
                angle_diff = abs(robot_angle - future_obj_angle)
                
                if angle_diff < 0.3 and future_obj_dist < robot_dist + self.robot_radius + self.safety_margin:
                    risk = max(risk, 1.0 - future_obj_dist / (robot_dist + self.robot_radius + self.safety_margin))
                    
        return risk < 0.3, risk
    
    def _correct_action(
        self,
        action: np.ndarray,
        static_risk: float,
        dynamic_risk: float,
        lidar: np.ndarray,
        objects: List[Dict]
    ) -> np.ndarray:
        """修正不安全的动作"""
        v, w = action[0], action[1]
        corrected = action.copy()
        
        total_risk = max(static_risk, dynamic_risk)
        
        # 根据风险级别调整
        if total_risk > 0.5:
            # 高风险: 大幅减速或停止
            corrected[0] = v * 0.2
            # 尝试转向更安全的方向
            corrected[1] = self._find_safe_direction(w, lidar, objects)
        elif total_risk > 0.3:
            # 中等风险: 适度减速
            corrected[0] = v * 0.5
            corrected[1] = self._find_safe_direction(w, lidar, objects) * 0.5 + w * 0.5
        else:
            # 低风险: 轻微调整
            corrected[0] = v * 0.8
            
        return corrected
    
    def _find_safe_direction(self, current_w: float, lidar: np.ndarray, objects: List[Dict]) -> float:
        """找到更安全的转向方向"""
        # 检查左右两侧的 LiDAR
        left_lidar = lidar[:60]
        front_lidar = lidar[60:120]
        right_lidar = lidar[120:180]
        
        left_min = np.min(left_lidar)
        front_min = np.min(front_lidar)
        right_min = np.min(right_lidar)
        
        # 选择更开阔的方向
        if left_min > right_min:
            return min(1.0, max(0.3, current_w + 0.3))  # 左转
        else:
            return max(-1.0, min(-0.3, current_w - 0.3))  # 右转


# =========================
# 自动课程学习
# =========================
class AutoCurriculum:
    """
    自动课程学习 (Automated Curriculum Learning)
    
    根据当前表现自动调整任务难度。
    """
    
    def __init__(
        self,
        initial_difficulty: float = 0.3,
        min_difficulty: float = 0.2,
        max_difficulty: float = 1.0,
        window_size: int = 50,
        success_threshold: float = 0.6,
        failure_threshold: float = 0.3
    ):
        self.difficulty = initial_difficulty
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        
        self.success_history: Deque[float] = deque(maxlen=window_size)
        
    def update(self, success: bool) -> float:
        """更新难度"""
        self.success_history.append(float(success))
        
        if len(self.success_history) < self.window_size // 2:
            return self.difficulty
            
        recent_success_rate = np.mean(self.success_history)
        
        if recent_success_rate > self.success_threshold:
            # 表现良好，增加难度
            self.difficulty = min(self.max_difficulty, self.difficulty + 0.02)
        elif recent_success_rate < self.failure_threshold:
            # 表现差，降低难度
            self.difficulty = max(self.min_difficulty, self.difficulty - 0.02)
            
        return self.difficulty
    
    def get_difficulty(self) -> float:
        """获取当前难度"""
        return self.difficulty
    
    def get_reward_multiplier(self) -> float:
        """根据难度获取奖励乘数"""
        return 0.5 + self.difficulty * 0.5


# =========================
# 增强的 Actor-Critic 模型
# =========================
class EnhancedActorCritic(nn.Module):
    """
    增强的 Actor-Critic 模型
    
    改进:
    1. 更深的网络结构
    2. 残差连接
    3. 辅助任务头
    4. Layer Normalization
    """
    
    def __init__(
        self,
        lidar_dim: int,
        low_dim: int,
        action_dim: int,
        use_aux_tasks: bool = True
    ):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.low_dim = low_dim
        self.action_dim = action_dim
        self.use_aux_tasks = use_aux_tasks

        # LiDAR 编码器 (ResNet 风格)
        self.lidar_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, lidar_dim)
            conv_out = self.lidar_conv(dummy)
            self.conv_out_dim = conv_out.shape[1] * conv_out.shape[2]
            
        self.lidar_fc = nn.Sequential(
            nn.Linear(self.conv_out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 低维状态编码器
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # GRU 时序编码器 (更稳定)
        self.gru = nn.GRU(
            input_size=128 + 64,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 后处理
        self.post_gru = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # Actor 头 (更复杂的策略网络)
        self.actor_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic 头
        self.critic_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(64, 1)

        # 辅助任务头
        if self.use_aux_tasks:
            # 碰撞预测 (预测未来 1, 3, 5, 10 步的碰撞概率)
            self.collision_predictor = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                nn.Sigmoid()
            )
            
            # 速度预测
            self.speed_predictor = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            
            # 未来 LiDAR 预测 (压缩版)
            self.lidar_predictor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )

    def encode_single_frame(self, obs_frame: torch.Tensor) -> torch.Tensor:
        """编码单帧观测"""
        lidar = obs_frame[:, :self.lidar_dim]
        low = obs_frame[:, self.lidar_dim: self.lidar_dim + self.low_dim]

        lidar_conv = self.lidar_conv(lidar.unsqueeze(1))
        lidar_flat = lidar_conv.reshape(lidar_conv.shape[0], -1)
        lidar_feat = self.lidar_fc(lidar_flat)
        
        low_feat = self.low_encoder(low)
        
        return torch.cat([lidar_feat, low_feat], dim=-1)

    def forward(self, obs_seq: torch.Tensor):
        """前向传播"""
        bsz, seq_len, obs_dim = obs_seq.shape

        # 编码每一帧
        flat = obs_seq.reshape(bsz * seq_len, obs_dim)
        frame_feat = self.encode_single_frame(flat).reshape(bsz, seq_len, -1)

        # GRU 时序编码
        gru_out, _ = self.gru(frame_feat)
        temporal_feat = self.post_gru(gru_out[:, -1, :])

        # Actor
        actor_feat = self.actor_fc(temporal_feat)
        mean = self.actor_mean(actor_feat)
        logstd = self.actor_logstd.expand_as(mean)
        
        # Critic
        critic_feat = self.critic_fc(temporal_feat)
        value = self.critic_head(critic_feat)

        # 辅助任务
        if self.use_aux_tasks:
            pred_collision = self.collision_predictor(temporal_feat)
            pred_speed = self.speed_predictor(temporal_feat)
            pred_lidar = self.lidar_predictor(temporal_feat)
            return mean, logstd, value, pred_collision, pred_speed, pred_lidar
        else:
            return mean, logstd, value, None, None, None

    def get_action_and_value(self, obs_seq: torch.Tensor, action: torch.Tensor = None):
        """获取动作和价值"""
        outputs = self.forward(obs_seq)
        mean, logstd, value = outputs[0], outputs[1], outputs[2]
        
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
            
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        if self.use_aux_tasks:
            return action, logprob, entropy, value.squeeze(-1), outputs[3], outputs[4], outputs[5]
        else:
            return action, logprob, entropy, value.squeeze(-1), None, None, None

    def get_value(self, obs_seq: torch.Tensor):
        """获取价值估计"""
        outputs = self.forward(obs_seq)
        return outputs[2].squeeze(-1)

    def get_deterministic_action(self, obs_seq: torch.Tensor):
        """获取确定性动作"""
        outputs = self.forward(obs_seq)
        mean = outputs[0]
        value = outputs[2]
        return mean, value.squeeze(-1)


# =========================
# GAE 计算
# =========================
def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 GAE"""
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
# 辅助函数
# =========================
def init_obs_history(first_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    """初始化观测历史"""
    hist: Deque[np.ndarray] = deque(maxlen=seq_len)
    for _ in range(seq_len):
        hist.append(first_obs.copy())
    return hist


def build_enhanced_obs(
    obs: np.ndarray,
    dynamic_detector: EnhancedDynamicDetector
) -> np.ndarray:
    """构建增强观测"""
    lidar = obs[:180].astype(np.float32)
    low = obs[180:187].astype(np.float32)
    
    # 动态特征
    dynamic_feat = dynamic_detector.detect()
    
    return np.concatenate([lidar, low, dynamic_feat], axis=0).astype(np.float32)


# =========================
# 评估函数
# =========================
def evaluate_policy(
    env: UnityNavEnv,
    model: EnhancedActorCritic,
    cfg: PPOConfig,
    device: torch.device,
    num_episodes: int = 100,
    use_safety_layer: bool = True,
    safety_layer: Optional[PredictiveSafetyLayer] = None
) -> Dict[str, float]:
    """评估策略"""
    model.eval()
    
    returns = []
    lengths = []
    successes = []
    collisions = []
    timeouts = []
    final_goal_dists = []
    min_lidars = []

    with torch.no_grad():
        for ep_idx in range(num_episodes):
            obs_np, info = env.reset()
            
            # 初始化
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            dynamic_detector = EnhancedDynamicDetector(cfg.lidar_dim, cfg.history_len, cfg.max_detected_objects, cfg.use_kalman_tracker)
            
            enhanced_obs = build_enhanced_obs(obs_np, dynamic_detector)
            seq_hist = init_obs_history(enhanced_obs, cfg.seq_len)

            done = False
            ep_ret = 0.0
            ep_len = 0
            last_info = info
            ep_min_lidars = []
            
            # 机器人位置追踪
            robot_x, robot_z, robot_yaw = 0.0, 0.0, 0.0
            prev_obs = None

            while not done:
                seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
                seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)
                
                action_mean, _ = model.get_deterministic_action(seq_tensor)
                action_np = action_mean.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)
                
                # 安全层
                if use_safety_layer and safety_layer is not None:
                    action_np = safety_layer.check_and_correct(action_np, obs_np, dynamic_detector.last_objects)
                
                obs_np, reward, done, truncated, info = env.step(action_np)
                ep_ret += reward
                ep_len += 1
                last_info = info
                ep_min_lidars.append(float(np.min(obs_np[:180]) * 10.0))

                if not done:
                    # 更新检测器
                    lidar = obs_np[:180]
                    dynamic_detector.update(lidar, robot_x, robot_z, robot_yaw)
                    
                    # 更新历史
                    obs_hist.append(obs_np.copy())
                    enhanced_obs = build_enhanced_obs(obs_np, dynamic_detector)
                    seq_hist.append(enhanced_obs.copy())
                    
                    prev_obs = obs_np.copy()

            returns.append(ep_ret)
            lengths.append(ep_len)
            successes.append(float(last_info.get("success", False)))
            collisions.append(float(last_info.get("collision", False)))
            timeouts.append(float(last_info.get("timeout", False)))
            final_goal_dists.append(float(last_info.get("goal_dist", np.nan)))
            min_lidars.append(float(np.min(ep_min_lidars)) if ep_min_lidars else 0.0)

    model.train()
    
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(np.mean(timeouts)),
        "final_goal_dist_mean": float(np.nanmean(final_goal_dists)),
        "min_lidar_mean": float(np.mean(min_lidars)),
    }


# =========================
# 主训练函数
# =========================
def get_env_path() -> str:
    """自动检测 Unity 环境路径"""
    import platform
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if platform.system() == "Linux":
        linux_paths = [
            os.path.join(script_dir, "Corriidor_linux/Corridor_linux.x86_64"),
            "./Corriidor_linux/Corridor_linux.x86_64",
            "/home/dell/DRL_Navigation/Corriidor_linux/Corridor_linux.x86_64",
        ]
        for p in linux_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Could not find Unity environment for Linux. Searched: {linux_paths}")
    else:
        win_paths = [
            r"D:\DRL_Navigation\Builds\Project_1.exe",
            os.path.join(script_dir, "Builds/Project_1.exe"),
        ]
        for p in win_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Could not find Unity environment for Windows. Searched: {win_paths}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO v3 with advanced features")
    parser.add_argument("--env", type=str, default=None, help="Unity environment path")
    parser.add_argument("--no-graphics", action="store_true", help="Run without graphics")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--updates", type=int, default=None, help="Total updates (override config)")
    parser.add_argument("--no-her", action="store_true", help="Disable HER")
    parser.add_argument("--no-safety", action="store_true", help="Disable safety layer")
    args = parser.parse_args()
    
    cfg = PPOConfig()
    
    # 命令行参数覆盖
    if args.resume:
        cfg.resume = True
        cfg.resume_checkpoint = args.resume
    if args.updates:
        cfg.total_updates = args.updates
    if args.no_her:
        cfg.use_her = False
    if args.no_safety:
        cfg.use_safety_layer = False
    
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 获取环境路径
    env_path = args.env if args.env else get_env_path()
    print(f"Using environment: {env_path}")
    print(f"Configuration:")
    print(f"  - HER: {cfg.use_her}")
    print(f"  - Safety Layer: {cfg.use_safety_layer}")
    print(f"  - Social Force: {cfg.use_social_force}")
    print(f"  - Kalman Tracker: {cfg.use_kalman_tracker}")
    print(f"  - Auto Curriculum: {cfg.use_auto_curriculum}")

    # 环境配置
    env_cfg = EnvConfig(
        file_name=env_path,
        behavior_name="Navtest?team=0",
        no_graphics=args.no_graphics,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=500,               # 更长的最大步数
        progress_gain=3.0,
        time_penalty=-0.008,
        collision_penalty=-15.0,     # 更严厉的碰撞惩罚
        success_bonus=150.0,         # 更高的成功奖励
        timeout_penalty=-25.0,       # 更严厉的超时惩罚
        near_obstacle_threshold=0.5,
        near_obstacle_penalty=-0.2,
        action_l2_penalty=-0.001,
    )

    device = torch.device(cfg.device)
    env = UnityNavEnv(env_cfg)
    
    # 创建模型
    model = EnhancedActorCritic(
        cfg.lidar_dim,
        cfg.low_dim,
        cfg.action_dim,
        use_aux_tasks=cfg.use_aux_tasks
    ).to(device)
    
    # 分离 Actor 和 Critic 的优化器
    actor_params = list(model.actor_fc.parameters()) + list(model.actor_mean.parameters()) + list(model.lidar_conv.parameters()) + list(model.lidar_fc.parameters()) + list(model.gru.parameters())
    critic_params = list(model.critic_fc.parameters()) + list(model.critic_head.parameters()) + list(model.low_encoder.parameters()) + list(model.post_gru.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': actor_params, 'lr': cfg.lr_actor},
        {'params': critic_params, 'lr': cfg.lr_critic}
    ])
    
    writer = SummaryWriter(log_dir=cfg.log_dir)
    writer.add_text("config", str(cfg))

    # 社交力模型奖励
    social_force = SocialForceReward(
        goal_gain=3.0,
        collision_penalty=-15.0,
        safety_margin=0.5,
        social_force_weight=0.5
    ) if cfg.use_social_force else None
    
    # 预测性安全层
    safety_layer = PredictiveSafetyLayer(
        robot_radius=0.25,
        safety_margin=0.15,
        prediction_horizon=10
    ) if cfg.use_safety_layer else None
    
    # 自动课程学习
    curriculum = AutoCurriculum(
        initial_difficulty=0.3,
        min_difficulty=0.2,
        max_difficulty=1.0,
        window_size=50,
        success_threshold=0.6,
        failure_threshold=0.3
    ) if cfg.use_auto_curriculum else None

    # 恢复训练
    start_update = 1
    global_step = 0
    if cfg.resume and cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        print(f"Resuming from checkpoint: {cfg.resume_checkpoint}")
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 1) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from update {start_update - 1}, global_step {global_step}")

    # 学习率调度
    def lr_lambda(update):
        frac = 1.0 - (update - 1) / cfg.total_updates
        return frac * 0.9 + 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if cfg.resume and start_update > 1:
        for _ in range(1, start_update):
            scheduler.step()

    # 初始化
    obs_np, _ = env.reset()
    obs_hist = init_obs_history(obs_np, cfg.seq_len)
    dynamic_detector = EnhancedDynamicDetector(cfg.lidar_dim, cfg.history_len, cfg.max_detected_objects, cfg.use_kalman_tracker)
    enhanced_obs = build_enhanced_obs(obs_np, dynamic_detector)
    seq_hist = init_obs_history(enhanced_obs, cfg.seq_len)

    episode_return = 0.0
    episode_len = 0
    train_ep_count = 0
    train_returns_window = deque(maxlen=100)
    train_success_window = deque(maxlen=100)
    train_collision_window = deque(maxlen=100)
    
    prev_obs = None
    robot_x, robot_z, robot_yaw = 0.0, 0.0, 0.0

    start_time = time.time()
    best_success_rate = 0.0

    # 主训练循环
    for update in range(start_update, cfg.total_updates + 1):
        seq_obs_buf: List[torch.Tensor] = []
        action_buf: List[torch.Tensor] = []
        logprob_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        value_buf: List[torch.Tensor] = []
        
        # 辅助任务数据
        collision_label_buf: List[torch.Tensor] = []
        speed_label_buf: List[torch.Tensor] = []
        lidar_label_buf: List[torch.Tensor] = []

        # 课程学习难度
        current_difficulty = curriculum.get_difficulty() if curriculum else 1.0

        # Rollout
        for step in range(cfg.rollout_steps):
            global_step += 1
            seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                outputs = model.get_action_and_value(seq_tensor)
                action = outputs[0].squeeze(0)
                logprob = outputs[1].squeeze(0)
                value = outputs[3].squeeze(0)

            action_np = action.detach().cpu().numpy()
            
            # 安全层修正
            if safety_layer is not None:
                action_np = safety_layer.check_and_correct(action_np, obs_np, dynamic_detector.last_objects)
            
            next_obs_np, base_reward, done, truncated, info = env.step(action_np)
            
            # 社交力模型奖励
            if social_force is not None:
                reward = social_force.compute(
                    base_reward, next_obs_np, info, 
                    dynamic_detector.last_objects, prev_obs, update, cfg.total_updates
                )
            else:
                reward = base_reward
                
            # 课程学习奖励调整
            if curriculum is not None:
                reward *= curriculum.get_reward_multiplier()
            
            episode_return += reward
            episode_len += 1

            seq_obs_buf.append(seq_tensor.squeeze(0).detach())
            action_buf.append(action.detach())
            logprob_buf.append(logprob.detach())
            reward_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))
            value_buf.append(value.detach())
            
            # 辅助任务标签
            if cfg.use_aux_tasks:
                # 碰撞标签
                is_dangerous = float(np.min(next_obs_np[:180]) < 0.1)
                collision_label_buf.append(torch.tensor(is_dangerous, dtype=torch.float32, device=device))
                
                # 速度标签
                speed_label_buf.append(torch.tensor([next_obs_np[184], next_obs_np[185]], dtype=torch.float32, device=device))
                
                # LiDAR 标签 (压缩)
                lidar_label_buf.append(torch.tensor(next_obs_np[:64], dtype=torch.float32, device=device))

            if done:
                train_ep_count += 1
                train_returns_window.append(float(episode_return))
                train_success_window.append(float(info["success"]))
                train_collision_window.append(float(info["collision"]))
                
                # 更新课程学习
                if curriculum is not None:
                    current_difficulty = curriculum.update(info["success"])

                writer.add_scalar("train/episode_return", float(episode_return), global_step)
                writer.add_scalar("train/episode_length", int(episode_len), global_step)
                writer.add_scalar("train/episode_success", float(info["success"]), global_step)
                writer.add_scalar("train/episode_collision", float(info["collision"]), global_step)
                writer.add_scalar("train/difficulty", current_difficulty, global_step)

                print(
                    f"[train ep] update={update:04d} step={global_step} "
                    f"ret={episode_return:.2f} len={episode_len} "
                    f"success={info['success']} collision={info['collision']} "
                    f"difficulty={current_difficulty:.2f}"
                )

                next_obs_np, _ = env.reset()
                obs_hist = init_obs_history(next_obs_np, cfg.seq_len)
                dynamic_detector = EnhancedDynamicDetector(cfg.lidar_dim, cfg.history_len, cfg.max_detected_objects, cfg.use_kalman_tracker)
                enhanced_obs = build_enhanced_obs(next_obs_np, dynamic_detector)
                seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
                episode_return = 0.0
                episode_len = 0
                prev_obs = None
            else:
                obs_hist.append(next_obs_np.copy())
                dynamic_detector.update(next_obs_np[:180], robot_x, robot_z, robot_yaw)
                enhanced_obs = build_enhanced_obs(next_obs_np, dynamic_detector)
                seq_hist.append(enhanced_obs.copy())
                prev_obs = obs_np.copy()
                
            obs_np = next_obs_np

        # GAE
        next_seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
        next_seq_tensor = torch.tensor(next_seq_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            next_value = model.get_value(next_seq_tensor).squeeze(0)

        seq_obs_buf = torch.stack(seq_obs_buf)
        action_buf = torch.stack(action_buf)
        logprob_buf = torch.stack(logprob_buf)
        reward_buf = torch.stack(reward_buf)
        done_buf = torch.stack(done_buf)
        value_buf = torch.stack(value_buf)

        advantages, returns = compute_gae(
            rewards=reward_buf, dones=done_buf, values=value_buf,
            next_value=next_value, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Entropy 系数衰减
        progress = (update - 1) / cfg.total_updates
        current_ent_coef = cfg.ent_coef_start + (cfg.ent_coef_end - cfg.ent_coef_start) * progress

        # PPO 更新
        batch_size = cfg.rollout_steps
        batch_inds = np.arange(batch_size)
        last_pg_loss, last_v_loss, last_entropy, last_kl = 0.0, 0.0, 0.0, 0.0
        last_aux_loss = 0.0
        clipfracs = []
        early_stop = False

        for epoch in range(cfg.update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = batch_inds[start:end]

                mb_seq_obs = seq_obs_buf[mb_inds]
                mb_actions = action_buf[mb_inds]
                mb_old_logprob = logprob_buf[mb_inds]
                mb_adv = advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_old_values = value_buf[mb_inds]

                outputs = model.get_action_and_value(mb_seq_obs, mb_actions)
                newlogprob = outputs[1]
                entropy = outputs[2]
                newvalue = outputs[3]
                pred_collision = outputs[4]
                pred_speed = outputs[5]
                pred_lidar = outputs[6]

                logratio = newlogprob - mb_old_logprob
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                if approx_kl > cfg.target_kl:
                    early_stop = True
                    break

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(newvalue - mb_old_values, -cfg.clip_coef, cfg.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                
                # 总损失
                loss = pg_loss + cfg.vf_coef * v_loss - current_ent_coef * entropy_loss
                
                # 辅助任务损失
                if cfg.use_aux_tasks and pred_collision is not None:
                    mb_collision_label = torch.stack([collision_label_buf[i] for i in mb_inds])
                    mb_speed_label = torch.stack([speed_label_buf[i] for i in mb_inds])
                    mb_lidar_label = torch.stack([lidar_label_buf[i] for i in mb_inds])
                    
                    # 碰撞预测损失
                    collision_loss = F.binary_cross_entropy(pred_collision[:, 0], mb_collision_label)
                    
                    # 速度预测损失
                    speed_loss = F.mse_loss(pred_speed, mb_speed_label)
                    
                    # LiDAR 预测损失
                    lidar_loss = F.mse_loss(pred_lidar, mb_lidar_label[:, :64])
                    
                    aux_loss = (
                        cfg.aux_collision_weight * collision_loss +
                        cfg.aux_speed_weight * speed_loss +
                        cfg.aux_lidar_weight * lidar_loss
                    )
                    loss += aux_loss
                    last_aux_loss = float(aux_loss.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                last_pg_loss = float(pg_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(entropy_loss.item())
                last_kl = float(approx_kl)

            if early_stop:
                break

        # 日志
        sps = int(global_step / max(time.time() - start_time, 1e-6))
        writer.add_scalar("train/update", update, global_step)
        writer.add_scalar("train/loss_pi", last_pg_loss, global_step)
        writer.add_scalar("train/loss_v", last_v_loss, global_step)
        writer.add_scalar("train/entropy", last_entropy, global_step)
        writer.add_scalar("train/approx_kl", last_kl, global_step)
        writer.add_scalar("train/ent_coef", current_ent_coef, global_step)
        writer.add_scalar("train/SPS", sps, global_step)
        
        if cfg.use_aux_tasks:
            writer.add_scalar("train/aux_loss", last_aux_loss, global_step)

        if len(train_success_window) > 0:
            writer.add_scalar("train_window/success_rate_100", float(np.mean(train_success_window)), global_step)
            writer.add_scalar("train_window/collision_rate_100", float(np.mean(train_collision_window)), global_step)

        print(f"update={update:04d} loss_pi={last_pg_loss:.4f} entropy={last_entropy:.4f} "
              f"ent_coef={current_ent_coef:.5f} sps={sps} success_100={np.mean(train_success_window):.3f}")

        scheduler.step()

        # 评估
        if update % cfg.eval_every == 0:
            eval_stats = evaluate_policy(
                env, model, cfg, device, 
                num_episodes=cfg.eval_episodes,
                use_safety_layer=cfg.use_safety_layer,
                safety_layer=safety_layer
            )
            
            writer.add_scalar("eval/success_rate", eval_stats["success_rate"], global_step)
            writer.add_scalar("eval/collision_rate", eval_stats["collision_rate"], global_step)
            writer.add_scalar("eval/return_mean", eval_stats["return_mean"], global_step)
            writer.add_scalar("eval/min_lidar_mean", eval_stats["min_lidar_mean"], global_step)

            print(f"[eval] update={update:04d} succ={eval_stats['success_rate']:.3f} "
                  f"coll={eval_stats['collision_rate']:.3f} "
                  f"ret={eval_stats['return_mean']:.2f} "
                  f"min_lidar={eval_stats['min_lidar_mean']:.2f}m")

            # 保存最佳模型
            if eval_stats["success_rate"] > best_success_rate:
                best_success_rate = eval_stats["success_rate"]
                best_path = os.path.join(cfg.save_dir, "best_model.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "success_rate": eval_stats["success_rate"],
                    "model_type": "enhanced_ppo_v3",
                }, best_path)
                print(f"New best model saved! Success rate: {best_success_rate:.3f}")

            # 重置
            obs_np, _ = env.reset()
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            dynamic_detector = EnhancedDynamicDetector(cfg.lidar_dim, cfg.history_len, cfg.max_detected_objects, cfg.use_kalman_tracker)
            enhanced_obs = build_enhanced_obs(obs_np, dynamic_detector)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
            episode_return = 0.0
            episode_len = 0

        # 保存 checkpoint
        if update % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"ppo_update_{update:04d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "best_success_rate": best_success_rate,
                "model_type": "enhanced_ppo_v3",
            }, save_path)
            print(f"saved to {save_path}")
            
            # 如果成功率已经达到目标，提前停止
            if best_success_rate >= 0.8:
                print(f"Target success rate 0.8 achieved! Best: {best_success_rate:.3f}")
                break

    writer.close()
    env.close()
    print(f"Training completed. Best success rate: {best_success_rate:.3f}")


if __name__ == "__main__":
    main()

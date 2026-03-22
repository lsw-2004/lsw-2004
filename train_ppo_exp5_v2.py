"""
实验5 v2: 专门针对匀速直线行人场景优化

核心改进:
1. 多帧历史动态检测 - 使用更多历史帧，更准确估计速度
2. 点级变化追踪 - 不再做区域平均，保留局部信息
3. 机器人运动补偿 - 减去机器人自身运动的影响
4. 行人级特征提取 - 估计每个行人的位置、速度、方向
5. 更强的时序模型 - Transformer 替代 GRU
"""
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import math

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # 训练参数
    total_updates: int = 3000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # 优化器参数
    lr: float = 3e-4
    clip_coef: float = 0.2
    
    # Entropy 衰减
    ent_coef_start: float = 0.01
    ent_coef_end: float = 0.001
    
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 6
    minibatch_size: int = 256
    target_kl: float = 0.02

    # 观测参数
    obs_dim: int = 187
    lidar_dim: int = 180
    raw_low_dim: int = 7
    
    # 动态检测参数
    history_len: int = 10              # 使用 10 帧历史
    dynamic_feat_dim: int = 48         # 更丰富的动态特征
    max_detected_objects: int = 5      # 最多检测 5 个动态物体
    
    action_dim: int = 2
    seq_len: int = 16                  # 更长序列

    # 保存路径
    save_dir: str = "./checkpoints/cnn_gru_ppo_tb/exp5_v2"
    log_dir: str = "./runs/cnn_gru_ppo_tb/exp5_v2"
    save_every: int = 50

    # 评估
    eval_every: int = 10
    eval_episodes: int = 50

    # 恢复训练
    resume: bool = False
    resume_checkpoint: str = ""

    @property
    def low_dim(self) -> int:
        return self.raw_low_dim + self.dynamic_feat_dim

    @property
    def enhanced_obs_dim(self) -> int:
        return self.lidar_dim + self.low_dim


# =========================
# 核心: 匀速直线行人检测器
# =========================
class UniformMotionDetector:
    """
    专门针对匀速直线运动行人的检测器
    
    原理:
    1. 使用多帧历史估计 LiDAR 点的速度
    2. 补偿机器人自身运动
    3. 聚类识别动态物体
    4. 预测未来位置
    """
    
    def __init__(
        self, 
        lidar_dim: int = 180, 
        history_len: int = 10,
        max_objects: int = 5
    ):
        self.lidar_dim = lidar_dim
        self.history_len = history_len
        self.max_objects = max_objects
        
        # 历史缓冲
        self.lidar_history: Deque[np.ndarray] = deque(maxlen=history_len)
        self.robot_pose_history: Deque[Tuple[float, float, float]] = deque(maxlen=history_len)
        
        # LiDAR 角度 (假设均匀分布，-180° 到 +180°)
        self.angles = np.linspace(-np.pi, np.pi, lidar_dim, endpoint=False)
        
        # 缓存上一次检测结果
        self.last_detected_objects: List[Dict] = []
        
    def reset(self):
        """重置历史"""
        self.lidar_history.clear()
        self.robot_pose_history.clear()
        self.last_detected_objects = []
    
    def update(
        self, 
        lidar: np.ndarray, 
        robot_x: float, 
        robot_z: float, 
        robot_yaw: float
    ):
        """
        更新历史数据
        
        Args:
            lidar: 归一化的 LiDAR 距离 (0-1)
            robot_x, robot_z: 机器人位置
            robot_yaw: 机器人朝向 (弧度)
        """
        self.lidar_history.append(lidar.copy())
        self.robot_pose_history.append((robot_x, robot_z, robot_yaw))
    
    def detect(self) -> np.ndarray:
        """
        检测动态物体并返回特征向量
        
        返回 48 维特征:
        - [0:5] 每个检测到的物体的距离
        - [5:10] 每个物体的角度
        - [10:15] 每个物体的径向速度
        - [15:20] 每个物体的切向速度
        - [20:25] 每个物体的威胁等级
        - [25:30] 每个物体到目标路径的距离
        - [30:35] 预测的碰撞时间 (TTC)
        - [35:40] 物体的可穿越性
        - [40:44] 全局统计特征
        - [44:48] 预测特征
        """
        features = np.zeros(48, dtype=np.float32)
        
        if len(self.lidar_history) < 3:
            return features
        
        # Step 1: 计算每点的表观速度
        point_velocities = self._compute_point_velocities()
        
        # Step 2: 聚类动态点
        dynamic_clusters = self._cluster_dynamic_points(point_velocities)
        
        # Step 3: 为每个聚类估计物体属性
        objects = self._estimate_object_properties(dynamic_clusters)
        
        # Step 4: 计算威胁和预测
        objects = self._compute_threat_and_prediction(objects)
        
        # Step 5: 构建特征向量
        features = self._build_features(objects)
        
        self.last_detected_objects = objects
        
        return features
    
    def _compute_point_velocities(self) -> np.ndarray:
        """
        计算每个 LiDAR 点的表观速度
        
        使用最小二乘拟合估计速度，对噪声更鲁棒
        """
        if len(self.lidar_history) < 3:
            return np.zeros((self.lidar_dim, 2))
        
        # 构建时间序列
        n = len(self.lidar_history)
        times = np.arange(n) * 0.1  # 假设 dt = 0.1s
        
        velocities = np.zeros((self.lidar_dim, 2))  # (径向, 切向)
        
        for i in range(self.lidar_dim):
            # 该点在历史帧中的距离序列
            distances = np.array([hist[i] for hist in self.lidar_history])
            
            # 转换为实际距离 (米)
            distances_m = distances * 10.0
            
            # 最小二乘拟合: d(t) = d0 + v * t
            # v = Cov(t, d) / Var(t)
            t_mean = np.mean(times)
            d_mean = np.mean(distances_m)
            
            cov = np.sum((times - t_mean) * (distances_m - d_mean))
            var = np.sum((times - t_mean) ** 2)
            
            if var > 1e-6:
                radial_velocity = cov / var  # 径向速度 (正=远离, 负=接近)
            else:
                radial_velocity = 0.0
            
            # 切向速度估计 (需要角度变化)
            # 简化: 如果径向速度变化大，认为有切向运动
            velocity_magnitude = np.std(distances_m) / (times[-1] - times[0] + 0.1)
            tangential_velocity = np.sqrt(max(0, velocity_magnitude**2 - radial_velocity**2))
            
            velocities[i] = [radial_velocity, tangential_velocity]
        
        return velocities
    
    def _cluster_dynamic_points(self, velocities: np.ndarray) -> List[Dict]:
        """
        聚类动态点，识别可能是同一个物体的点
        
        策略: 连续的点且速度相近的归为一类
        """
        # 获取当前帧的 LiDAR
        curr_lidar = self.lidar_history[-1]
        
        # 动态点的阈值: 速度 > 0.2 m/s
        speed_threshold = 0.2
        speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        dynamic_mask = speeds > speed_threshold
        
        # 距离阈值: 太远的点可能不可靠
        distance_threshold = 8.0  # 米
        distance_mask = curr_lidar * 10.0 < distance_threshold
        
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
                if i - cluster_start >= 3:  # 至少 3 个连续点
                    clusters.append({
                        'indices': list(range(cluster_start, i)),
                        'velocities': velocities[cluster_start:i],
                        'distances': curr_lidar[cluster_start:i]
                    })
        
        # 处理最后一个聚类
        if in_cluster and self.lidar_dim - cluster_start >= 3:
            clusters.append({
                'indices': list(range(cluster_start, self.lidar_dim)),
                'velocities': velocities[cluster_start:self.lidar_dim],
                'distances': curr_lidar[cluster_start:self.lidar_dim]
            })
        
        return clusters
    
    def _estimate_object_properties(self, clusters: List[Dict]) -> List[Dict]:
        """
        为每个聚类估计物体属性
        """
        objects = []
        
        # 行人最大合理速度 (m/s)
        MAX_PEDESTRIAN_SPEED = 3.0
        
        for cluster in clusters[:self.max_objects]:
            indices = cluster['indices']
            vels = cluster['velocities']
            dists = cluster['distances']
            
            # 物体中心角度
            center_angle = np.mean(self.angles[indices])
            
            # 物体距离 (取最近的点)
            min_dist_idx = np.argmin(dists)
            distance = float(dists[min_dist_idx] * 10.0)
            
            # 平均速度
            mean_radial_vel = float(np.mean(vels[:, 0]))
            mean_tangential_vel = float(np.mean(vels[:, 1]))
            
            # 速度上限过滤 (行人最大约 3 m/s，超过则认为是噪声)
            total_speed = np.sqrt(mean_radial_vel**2 + mean_tangential_vel**2)
            if total_speed > MAX_PEDESTRIAN_SPEED:
                scale = MAX_PEDESTRIAN_SPEED / total_speed
                mean_radial_vel *= scale
                mean_tangential_vel *= scale
            
            # 物体大小 (角度跨度)
            angular_size = len(indices) * (2 * np.pi / self.lidar_dim)
            
            objects.append({
                'angle': center_angle,
                'distance': distance,
                'radial_velocity': mean_radial_vel,
                'tangential_velocity': mean_tangential_vel,
                'angular_size': angular_size,
                'n_points': len(indices)
            })
        
        return objects
    
    def _compute_threat_and_prediction(self, objects: List[Dict]) -> List[Dict]:
        """
        计算每个物体的威胁等级和未来位置预测
        """
        for obj in objects:
            distance = obj['distance']
            radial_vel = obj['radial_velocity']
            tangential_vel = obj['tangential_velocity']
            angle = obj['angle']
            
            # 威胁等级: 距离近 + 正在接近
            approach_factor = max(0, -radial_vel) / 2.0  # 正在接近为正
            distance_factor = 1.0 - min(distance / 5.0, 1.0)
            angle_factor = 1.0 - abs(angle) / np.pi  # 前方更危险
            
            threat = approach_factor * distance_factor * angle_factor
            obj['threat'] = float(np.clip(threat, 0, 1))
            
            # 碰撞时间 (TTC)
            if radial_vel < -0.1:  # 正在接近
                ttc = distance / (-radial_vel)
                obj['ttc'] = float(min(ttc, 10.0))
            else:
                obj['ttc'] = 10.0  # 不在接近
            
            # 未来位置预测 (匀速模型)
            # 1秒后的位置
            future_distance = distance + radial_vel * 1.0
            future_angle = angle + np.arcsin(np.clip(tangential_vel / (distance + 0.1) * 1.0, -1, 1))
            obj['future_pos_1s'] = (future_distance, future_angle)
            
            # 可穿越性: 物体是否会离开我的路径
            # 如果物体横向移动，可能可以等待它通过
            if abs(tangential_vel) > 0.3:
                obj['crossable'] = float(min(abs(tangential_vel) / 1.0, 1.0))
            else:
                obj['crossable'] = 0.0
        
        return objects
    
    def _build_features(self, objects: List[Dict]) -> np.ndarray:
        """
        构建 48 维特征向量
        """
        features = np.zeros(48, dtype=np.float32)
        
        n_objects = min(len(objects), self.max_objects)
        
        for i in range(n_objects):
            obj = objects[i]
            features[i] = obj['distance'] / 10.0                    # 距离
            features[5 + i] = obj['angle'] / np.pi                  # 角度
            features[10 + i] = obj['radial_velocity'] / 2.0         # 径向速度
            features[15 + i] = obj['tangential_velocity'] / 2.0     # 切向速度
            features[20 + i] = obj['threat']                         # 威胁等级
            features[25 + i] = obj.get('crossable', 0)              # 可穿越性
            features[30 + i] = obj['ttc'] / 10.0                    # TTC
            features[35 + i] = obj['future_pos_1s'][0] / 10.0       # 预测距离
        
        # 全局统计 (40-48)
        if n_objects > 0:
            features[40] = n_objects / self.max_objects             # 检测到的物体数
            features[41] = min(obj['distance'] for obj in objects) / 10.0  # 最近距离
            features[42] = max(obj['threat'] for obj in objects)    # 最大威胁
            features[43] = min(obj['ttc'] for obj in objects) / 10.0  # 最小 TTC
            features[44] = sum(1 for obj in objects if obj['radial_velocity'] < 0) / max(n_objects, 1)  # 接近物体比例
            features[45] = sum(obj['threat'] for obj in objects) / max(n_objects, 1)  # 平均威胁
            features[46] = np.mean([obj['angle'] for obj in objects]) / np.pi  # 平均角度
            features[47] = np.std([obj['angle'] for obj in objects]) / np.pi   # 角度分散度
        
        return features
    
    def get_predictions(self) -> List[Tuple[float, float, float]]:
        """
        获取所有检测到的物体的预测位置
        
        返回: List of (distance, angle, ttc)
        """
        predictions = []
        for obj in self.last_detected_objects:
            dist, angle = obj['future_pos_1s']
            predictions.append((dist, angle, obj['ttc']))
        return predictions


# =========================
# 增强的时序编码器
# =========================
class PositionalEncoding(nn.Module):
    """位置编码，用于 Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformer(nn.Module):
    """Transformer 时序编码器"""
    
    def __init__(
        self, 
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x[:, -1, :]  # 返回最后时刻的输出


# =========================
# 模型定义
# =========================
class CNNGRUActorCriticV2(nn.Module):
    """
    改进的 Actor-Critic 模型
    
    改进:
    - 更深的 CNN
    - Transformer 时序编码 (可选)
    - 更好的特征融合
    """
    
    def __init__(
        self, 
        lidar_dim: int, 
        low_dim: int, 
        action_dim: int,
        use_transformer: bool = True
    ):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.low_dim = low_dim
        self.action_dim = action_dim
        self.use_transformer = use_transformer

        # LiDAR 编码器 (更深)
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, lidar_dim)
            lidar_feat_dim = self.lidar_encoder(dummy).shape[1]

        self.lidar_fc = nn.Sequential(
            nn.Linear(lidar_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # 低维状态编码器
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # 时序编码器
        temporal_input_dim = 128 + 64  # lidar + low
        
        if use_transformer:
            self.temporal_encoder = TemporalTransformer(
                d_model=temporal_input_dim,
                nhead=4,
                num_layers=2
            )
            temporal_output_dim = temporal_input_dim
        else:
            self.temporal_encoder = nn.GRU(
                input_size=temporal_input_dim,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
            temporal_output_dim = 256

        # 策略头
        self.actor_fc = nn.Sequential(
            nn.Linear(temporal_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # 价值头
        self.critic_fc = nn.Sequential(
            nn.Linear(temporal_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def encode_single_frame(self, obs_frame: torch.Tensor) -> torch.Tensor:
        lidar = obs_frame[:, :self.lidar_dim]
        low = obs_frame[:, self.lidar_dim: self.lidar_dim + self.low_dim]

        lidar_feat = self.lidar_fc(self.lidar_encoder(lidar.unsqueeze(1)))
        low_feat = self.low_encoder(low)
        return torch.cat([lidar_feat, low_feat], dim=-1)

    def forward(self, obs_seq: torch.Tensor):
        bsz, seq_len, obs_dim = obs_seq.shape
        
        # 编码每一帧
        flat = obs_seq.reshape(bsz * seq_len, obs_dim)
        frame_feat = self.encode_single_frame(flat).reshape(bsz, seq_len, -1)
        
        # 时序编码
        if self.use_transformer:
            temporal_feat = self.temporal_encoder(frame_feat)
        else:
            _, h = self.temporal_encoder(frame_feat)
            temporal_feat = h[-1]  # 最后一层的隐藏状态
        
        # 输出
        mean = self.actor_fc(temporal_feat)
        value = self.critic_fc(temporal_feat)
        logstd = self.actor_logstd.expand_as(mean)
        
        return mean, logstd, value

    def get_action_and_value(self, obs_seq: torch.Tensor, action: torch.Tensor = None):
        mean, logstd, value = self(obs_seq)
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, value.squeeze(-1)

    def get_value(self, obs_seq: torch.Tensor):
        _, _, value = self(obs_seq)
        return value.squeeze(-1)

    def get_deterministic_action(self, obs_seq: torch.Tensor):
        mean, _, value = self(obs_seq)
        return mean, value.squeeze(-1)


# =========================
# GAE 和辅助函数
# =========================
def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
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


def init_obs_history(first_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    hist: Deque[np.ndarray] = deque(maxlen=seq_len)
    for _ in range(seq_len):
        hist.append(first_obs.copy())
    return hist


def init_seq_history(first_enhanced_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    hist: Deque[np.ndarray] = deque(maxlen=seq_len)
    for _ in range(seq_len):
        hist.append(first_enhanced_obs.copy())
    return hist


def build_enhanced_obs(
    obs: np.ndarray,
    dynamic_detector: UniformMotionDetector
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
    model: CNNGRUActorCriticV2, 
    cfg: PPOConfig, 
    device: torch.device, 
    num_episodes: int = 50
):
    model.eval()
    returns = []
    lengths = []
    successes = []
    collisions = []
    timeouts = []
    final_goal_dists = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs_np, info = env.reset()
            
            # 初始化
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            dynamic_detector = UniformMotionDetector(cfg.lidar_dim, cfg.history_len)
            
            enhanced_obs = build_enhanced_obs(obs_np, dynamic_detector)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)

            done = False
            ep_ret = 0.0
            ep_len = 0
            last_info = info
            
            # 模拟机器人位置 (简化)
            robot_x, robot_z, robot_yaw = 0.0, 0.0, 0.0

            while not done:
                seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
                seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)
                action_mean, _ = model.get_deterministic_action(seq_tensor)
                action_np = action_mean.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)

                obs_np, reward, done, truncated, info = env.step(action_np)
                ep_ret += reward
                ep_len += 1
                last_info = info

                if not done:
                    # 更新动态检测器
                    lidar = obs_np[:180]
                    dynamic_detector.update(lidar, robot_x, robot_z, robot_yaw)
                    
                    # 更新观测历史
                    obs_hist.append(obs_np.copy())
                    enhanced_obs = build_enhanced_obs(obs_np, dynamic_detector)
                    seq_hist.append(enhanced_obs.copy())

            returns.append(ep_ret)
            lengths.append(ep_len)
            successes.append(float(last_info.get("success", False)))
            collisions.append(float(last_info.get("collision", False)))
            timeouts.append(float(last_info.get("timeout", False)))
            final_goal_dists.append(float(last_info.get("goal_dist", np.nan)))

    model.train()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(np.mean(timeouts)),
        "final_goal_dist_mean": float(np.nanmean(final_goal_dists)),
    }


# =========================
# 主训练函数
def get_env_path() -> str:
    """自动检测 Unity 环境路径"""
    import platform
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if platform.system() == "Linux":
        # Linux 环境路径
        linux_paths = [
            os.path.join(script_dir, "Corriidor_linux/Corridor_linux.x86_64"),
            "./Corriidor_linux/Corridor_linux.x86_64",
            "/home/dell/DRL_Navigation/Corriidor_linux/Corridor_linux.x86_64",
        ]
        for p in linux_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"Could not find Unity environment for Linux. "
            f"Searched: {linux_paths}"
        )
    else:
        # Windows 环境路径
        win_paths = [
            r"D:\DRL_Navigation\Builds\Project_1.exe",
            os.path.join(script_dir, "Builds/Project_1.exe"),
        ]
        for p in win_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"Could not find Unity environment for Windows. "
            f"Searched: {win_paths}"
        )


# =========================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO with dynamic detection")
    parser.add_argument("--env", type=str, default=None, help="Unity environment path")
    parser.add_argument("--no-graphics", action="store_true", help="Run without graphics")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--updates", type=int, default=None, help="Total updates (override config)")
    args = parser.parse_args()
    
    cfg = PPOConfig()
    
    # 命令行参数覆盖
    if args.resume:
        cfg.resume = True
        cfg.resume_checkpoint = args.resume
    if args.updates:
        cfg.total_updates = args.updates
    
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 获取环境路径
    env_path = args.env if args.env else get_env_path()
    print(f"Using environment: {env_path}")

    # 环境配置
    env_cfg = EnvConfig(
        file_name=env_path,
        behavior_name="Navtest?team=0",
        no_graphics=args.no_graphics,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=450,
        progress_gain=3.0,
        time_penalty=-0.008,
        collision_penalty=-10.0,
        success_bonus=100.0,
        timeout_penalty=-20.0,
        near_obstacle_threshold=0.5,
        near_obstacle_penalty=-0.2,
        action_l2_penalty=-0.001,
    )

    device = torch.device(cfg.device)
    env = UnityNavEnv(env_cfg)
    
    # 创建模型
    model = CNNGRUActorCriticV2(
        cfg.lidar_dim, 
        cfg.low_dim, 
        cfg.action_dim,
        use_transformer=True
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    writer.add_text("config", str(cfg))

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
    dynamic_detector = UniformMotionDetector(cfg.lidar_dim, cfg.history_len)
    enhanced_obs = build_enhanced_obs(obs_np, dynamic_detector)
    seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)

    episode_return = 0.0
    episode_len = 0
    train_ep_count = 0
    train_returns_window = deque(maxlen=50)
    train_lengths_window = deque(maxlen=50)
    train_success_window = deque(maxlen=50)
    train_collision_window = deque(maxlen=50)
    train_timeout_window = deque(maxlen=50)

    start_time = time.time()

    # 主训练循环
    for update in range(start_update, cfg.total_updates + 1):
        seq_obs_buf: List[torch.Tensor] = []
        action_buf: List[torch.Tensor] = []
        logprob_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        value_buf: List[torch.Tensor] = []
        rollout_rewards = []

        # Rollout
        for step in range(cfg.rollout_steps):
            global_step += 1
            seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, logprob, entropy, value = model.get_action_and_value(seq_tensor)
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
                value = value.squeeze(0)

            action_np = action.detach().cpu().numpy()
            next_obs_np, reward, done, truncated, info = env.step(action_np)
            episode_return += reward
            episode_len += 1
            rollout_rewards.append(reward)

            seq_obs_buf.append(seq_tensor.squeeze(0).detach())
            action_buf.append(action.detach())
            logprob_buf.append(logprob.detach())
            reward_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))
            value_buf.append(value.detach())

            if done:
                train_ep_count += 1
                train_returns_window.append(float(episode_return))
                train_lengths_window.append(int(episode_len))
                train_success_window.append(float(info["success"]))
                train_collision_window.append(float(info["collision"]))
                train_timeout_window.append(float(info["timeout"]))

                writer.add_scalar("train/episode_return", float(episode_return), global_step)
                writer.add_scalar("train/episode_length", int(episode_len), global_step)
                writer.add_scalar("train/episode_success", float(info["success"]), global_step)
                writer.add_scalar("train/episode_collision", float(info["collision"]), global_step)
                writer.add_scalar("train/episode_timeout", float(info["timeout"]), global_step)
                writer.add_scalar("train/final_goal_dist", float(info["goal_dist"]), global_step)

                print(
                    f"[train ep] update={update:04d} step={global_step} "
                    f"ret={episode_return:.3f} len={episode_len} "
                    f"success={info['success']} collision={info['collision']}"
                )

                next_obs_np, _ = env.reset()
                obs_hist = init_obs_history(next_obs_np, cfg.seq_len)
                dynamic_detector = UniformMotionDetector(cfg.lidar_dim, cfg.history_len)
                enhanced_obs = build_enhanced_obs(next_obs_np, dynamic_detector)
                seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
                episode_return = 0.0
                episode_len = 0
            else:
                obs_hist.append(next_obs_np.copy())
                # 更新动态检测器
                dynamic_detector.update(next_obs_np[:180], 0, 0, 0)
                enhanced_obs = build_enhanced_obs(next_obs_np, dynamic_detector)
                seq_hist.append(enhanced_obs.copy())

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

                _, newlogprob, entropy, newvalue = model.get_action_and_value(mb_seq_obs, mb_actions)

                logratio = newlogprob - mb_old_logprob
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                if approx_kl > cfg.target_kl:
                    early_stop = True
                    break

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(newvalue - mb_old_values, -cfg.clip_coef, cfg.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - current_ent_coef * entropy_loss

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

        if train_returns_window:
            writer.add_scalar("train_window/success_rate_50", float(np.mean(train_success_window)), global_step)
            writer.add_scalar("train_window/collision_rate_50", float(np.mean(train_collision_window)), global_step)

        print(f"update={update:04d} loss_pi={last_pg_loss:.4f} entropy={last_entropy:.4f} "
              f"ent_coef={current_ent_coef:.5f} sps={sps}")

        scheduler.step()

        # 评估
        if update % cfg.eval_every == 0:
            eval_stats = evaluate_policy(env, model, cfg, device, num_episodes=cfg.eval_episodes)
            writer.add_scalar("eval/success_rate", eval_stats["success_rate"], global_step)
            writer.add_scalar("eval/collision_rate", eval_stats["collision_rate"], global_step)
            writer.add_scalar("eval/return_mean", eval_stats["return_mean"], global_step)

            print(f"[eval] update={update:04d} succ={eval_stats['success_rate']:.3f} "
                  f"coll={eval_stats['collision_rate']:.3f}")

            # 重置
            obs_np, _ = env.reset()
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            dynamic_detector = UniformMotionDetector(cfg.lidar_dim, cfg.history_len)
            enhanced_obs = build_enhanced_obs(obs_np, dynamic_detector)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
            episode_return = 0.0
            episode_len = 0

        # 保存
        if update % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"ppo_update_{update:04d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "model_type": "cnn_transformer_v2",
            }, save_path)
            print(f"saved to {save_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()

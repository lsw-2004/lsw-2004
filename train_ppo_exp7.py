"""
实验7: 基于 Exp6，LiDAR 维度升级为 360°

核心改动:
1. LiDAR 维度从 180 维升级到 360 维 (全向感知)
2. Unity 端已导出 360° LiDAR 环境
3. 其他配置与 Exp6 完全一致

观测结构: LiDAR(360) + raw_low(7) + static_feat(6) + ped_feat(5*8=40) = 413 维
"""
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from unity_env import UnityNavEnv, EnvConfig


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

    total_updates: int = 3000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    clip_coef: float = 0.2
    ent_coef: float = 0.002  # 降低探索，更稳定
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 6
    minibatch_size: int = 256
    target_kl: float = 0.02  # 更保守更新

    obs_dim: int = 367           # Unity 输出: 360 LiDAR + 7 low-dim
    lidar_dim: int = 360         # 360° LiDAR
    raw_low_dim: int = 7
    action_dim: int = 2
    seq_len: int = 8

    # 行人预测参数
    lidar_history_len: int = 8       # LiDAR 历史帧数
    pedestrian_max_num: int = 5      # 最多检测多少个行人
    pedestrian_feat_dim: int = 8     # 每个行人的特征维度
    prediction_horizon: float = 1.5  # 预测未来多少秒

    # 静态障碍物风险特征 (新增)
    static_feat_dim: int = 6         # 静态障碍物特征维度

    save_dir: str = "./checkpoints/cnn_gru_ppo_tb/exp7"
    log_dir: str = "./runs/cnn_gru_ppo_tb/exp7"
    save_every: int = 50

    eval_every: int = 10
    eval_episodes: int = 50

    save_best: bool = True
    best_success_rate: float = 0.0

    resume: bool = False
    resume_checkpoint: str = ""

    @property
    def low_dim(self) -> int:
        # raw_low_dim(7) + 静态特征(6) + 行人特征 (max_num * feat_dim)
        return self.raw_low_dim + self.static_feat_dim + self.pedestrian_max_num * self.pedestrian_feat_dim

    @property
    def enhanced_obs_dim(self) -> int:
        return self.lidar_dim + self.low_dim


# =========================
# 行人检测与预测模块
# =========================
class PedestrianPredictor:
    """
    从 LiDAR 历史帧检测行人并预测未来位置
    
    原理:
    1. 对比相邻帧 LiDAR，找出移动的点 (动态障碍物)
    2. 对动态点聚类，每个簇是一个行人
    3. 估计每个行人的速度 (匀速假设)
    4. 预测未来位置
    """
    
    def __init__(self, lidar_dim: int = 360, max_pedestrians: int = 5,
                 prediction_horizon: float = 1.5, dt: float = 0.1):
        self.lidar_dim = lidar_dim
        self.max_pedestrians = max_pedestrians
        self.prediction_horizon = prediction_horizon
        self.dt = dt  # 帧间隔 (秒)
        
        # LiDAR 角度 - 360° 全向
        self.angles = np.linspace(-np.pi, np.pi, lidar_dim, endpoint=False)
        
        # 存储历史检测到的行人
        self.pedestrian_tracks: Dict[int, Dict] = {}  # id -> {positions, velocities}
        self.next_id = 0
    
    def detect_dynamic_points(self, lidar_history: Deque[np.ndarray], 
                               velocity_threshold: float = 0.1) -> np.ndarray:
        """
        检测动态点 (移动的 LiDAR 点)
        
        返回: (N, 4) 数组 [x, y, vx, vy]
        """
        if len(lidar_history) < 3:
            return np.zeros((0, 4), dtype=np.float32)
        
        # 取最近 3 帧
        curr_lidar = lidar_history[-1][:self.lidar_dim].astype(np.float32)
        prev_lidar = lidar_history[-2][:self.lidar_dim].astype(np.float32)
        prev2_lidar = lidar_history[-3][:self.lidar_dim].astype(np.float32)
        
        # 计算点的位置变化
        # LiDAR 数据是距离，需要转换为 x, y
        def lidar_to_xy(ranges):
            x = ranges * np.cos(self.angles)
            y = ranges * np.sin(self.angles)
            return x, y
        
        curr_x, curr_y = lidar_to_xy(curr_lidar)
        prev_x, prev_y = lidar_to_xy(prev_lidar)
        prev2_x, prev2_y = lidar_to_xy(prev2_lidar)
        
        # 计算速度 (两点差分)
        vx = (curr_x - prev_x) / self.dt
        vy = (curr_y - prev_y) / self.dt
        velocity_mag = np.sqrt(vx**2 + vy**2)
        
        # 筛选动态点: 速度大于阈值且距离有效
        valid_mask = (velocity_mag > velocity_threshold) & (curr_lidar < 5.0) & (curr_lidar > 0.1)
        
        # 同时检查连续两帧的一致性 (避免噪声)
        vx_prev = (prev_x - prev2_x) / self.dt
        vy_prev = (prev_y - prev2_y) / self.dt
        vel_mag_prev = np.sqrt(vx_prev**2 + vy_prev**2)
        consistent_mask = (vel_mag_prev > velocity_threshold * 0.5) & valid_mask
        
        # 合并
        dynamic_mask = valid_mask & consistent_mask
        
        if not np.any(dynamic_mask):
            # 退而求其次，只用当前帧
            dynamic_mask = valid_mask
        
        dynamic_x = curr_x[dynamic_mask]
        dynamic_y = curr_y[dynamic_mask]
        dynamic_vx = vx[dynamic_mask]
        dynamic_vy = vy[dynamic_mask]
        
        return np.column_stack([dynamic_x, dynamic_y, dynamic_vx, dynamic_vy])
    
    def cluster_pedestrians(self, dynamic_points: np.ndarray, 
                            cluster_radius: float = 0.8) -> List[Dict]:
        """
        简单聚类: 将靠近的动态点归为同一个行人
        
        返回: List of {position, velocity, point_count}
        """
        if len(dynamic_points) == 0:
            return []
        
        pedestrians = []
        used = np.zeros(len(dynamic_points), dtype=bool)
        
        for i in range(len(dynamic_points)):
            if used[i]:
                continue
            
            # 找所有距离这个点足够近的点
            distances = np.sqrt(
                (dynamic_points[:, 0] - dynamic_points[i, 0])**2 +
                (dynamic_points[:, 1] - dynamic_points[i, 1])**2
            )
            cluster_mask = (distances < cluster_radius) & (~used)
            
            if np.sum(cluster_mask) < 2:  # 至少 2 个点才算行人
                continue
            
            # 计算簇的质心和平均速度
            cluster_points = dynamic_points[cluster_mask]
            position = np.mean(cluster_points[:, :2], axis=0)
            velocity = np.mean(cluster_points[:, 2:4], axis=0)
            point_count = len(cluster_points)
            
            pedestrians.append({
                'position': position,
                'velocity': velocity,
                'point_count': point_count,
            })
            
            used[cluster_mask] = True
        
        # 按距离排序 (近的优先)
        pedestrians.sort(key=lambda p: np.linalg.norm(p['position']))
        
        return pedestrians[:self.max_pedestrians]
    
    def predict_future_positions(self, pedestrian: Dict) -> np.ndarray:
        """
        预测行人未来位置 (匀速直线假设)
        
        返回: (T, 2) 数组，未来 T 个时间步的位置
        """
        pos = pedestrian['position']
        vel = pedestrian['velocity']
        
        # 预测未来几个时间步
        t_steps = np.array([0.3, 0.6, 1.0, 1.5])  # 预测 0.3s, 0.6s, 1.0s, 1.5s
        future_positions = pos + np.outer(t_steps, vel)
        
        return future_positions
    
    def compute_risk_features(self, pedestrians: List[Dict], 
                               robot_pos: np.ndarray = np.zeros(2)) -> np.ndarray:
        """
        计算风险特征
        
        返回: (max_pedestrians, 8) 数组
        每个行人 8 维特征:
        - [0,1]: 相对位置 (x, y)
        - [2,3]: 速度 (vx, vy)
        - [4]: 距离
        - [5]: 速度大小
        - [6]: 最近预测点的距离
        - [7]: 碰撞风险分数 (0-1)
        """
        features = np.zeros((self.max_pedestrians, 8), dtype=np.float32)
        
        for i, ped in enumerate(pedestrians):
            if i >= self.max_pedestrians:
                break
            
            pos = ped['position']
            vel = ped['velocity']
            
            # 相对位置
            rel_pos = pos - robot_pos
            features[i, 0] = rel_pos[0]
            features[i, 1] = rel_pos[1]
            
            # 速度
            features[i, 2] = vel[0]
            features[i, 3] = vel[1]
            
            # 距离
            dist = np.linalg.norm(rel_pos)
            features[i, 4] = dist
            
            # 速度大小
            speed = np.linalg.norm(vel)
            features[i, 5] = speed
            
            # 预测未来位置，找最近距离
            future_pos = self.predict_future_positions(ped)
            future_dists = np.linalg.norm(future_pos - robot_pos, axis=1)
            min_future_dist = np.min(future_dists)
            features[i, 6] = min_future_dist
            
            # 碰撞风险 (距离越近、速度越快、朝向机器人 = 风险越高)
            if dist > 0.1:
                # 行人朝向
                ped_heading = np.arctan2(vel[1], vel[0])
                # 行人到机器人的方向
                to_robot = np.arctan2(-rel_pos[1], -rel_pos[0])
                # 角度差
                angle_diff = abs(ped_heading - to_robot)
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                
                # 风险分数
                distance_risk = 1.0 / (dist + 0.5)
                heading_risk = max(0, np.cos(angle_diff))  # 朝向机器人时高
                speed_risk = min(speed / 2.0, 1.0)  # 速度快时高
                
                risk = distance_risk * (0.3 + 0.7 * heading_risk) * speed_risk
                features[i, 7] = min(risk, 1.0)
        
        return features

    def compute_static_features(self, lidar: np.ndarray) -> np.ndarray:
        """
        计算静态障碍物风险特征 (6维) - 适配 360° LiDAR
        
        特征:
        - [0]: 全局最小距离
        - [1]: 前方 90° 最小距离
        - [2]: 左侧 90° 最小距离
        - [3]: 右侧 90° 最小距离
        - [4]: 最近障碍物角度 (归一化 -1 到 1)
        - [5]: 危险程度分数 (0-1)
        """
        features = np.zeros(6, dtype=np.float32)
        
        # 360° LiDAR 分区域 (每个区域 90° = 90 个点)
        # 角度范围: -180° 到 +180°，前方为 0°
        # rear:     -180° to -90°  (索引 0-89)
        # left:     -90° to 0°     (索引 90-179)
        # front:    0° to 90°      (索引 180-269)
        # right:    90° to 180°    (索引 270-359)
        
        rear = slice(0, 90)
        left = slice(90, 180)
        front = slice(180, 270)
        right = slice(270, 360)
        
        rear_lidar = lidar[rear]
        left_lidar = lidar[left]
        front_lidar = lidar[front]
        right_lidar = lidar[right]
        
        # 全局最小距离
        min_dist = float(np.min(lidar))
        features[0] = min_dist
        
        # 各区域最小距离
        features[1] = float(np.min(front_lidar))
        features[2] = float(np.min(left_lidar))
        features[3] = float(np.min(right_lidar))
        
        # 最近障碍物角度 (归一化到 -1 到 1)
        min_idx = int(np.argmin(lidar))
        # 索引 0-359 映射到角度 -180° 到 +180°
        min_angle = (min_idx - 180) / 180.0
        features[4] = min_angle
        
        # 危险程度分数
        front_risk = 1.0 / (features[1] + 0.3)
        global_risk = 1.0 / (min_dist + 0.3)
        features[5] = min(0.7 * global_risk + 0.3 * front_risk, 1.0)
        
        return features
    
    def update(self, lidar_history: Deque[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        主函数: 从 LiDAR 历史更新行人预测和静态障碍物特征
        
        返回: 
            - ped_features: (max_pedestrians, 8) 行人风险特征
            - static_features: (6,) 静态障碍物特征
        """
        # 1. 检测动态点
        dynamic_points = self.detect_dynamic_points(lidar_history)
        
        # 2. 聚类成行人
        pedestrians = self.cluster_pedestrians(dynamic_points)
        
        # 3. 计算行人风险特征
        ped_features = self.compute_risk_features(pedestrians)
        
        # 4. 计算静态障碍物特征
        curr_lidar = lidar_history[-1][:self.lidar_dim].astype(np.float32)
        static_features = self.compute_static_features(curr_lidar)
        
        return ped_features, static_features


def build_enhanced_obs(obs_hist: Deque[np.ndarray], 
                        lidar_history: Deque[np.ndarray],
                        pedestrian_predictor: PedestrianPredictor,
                        cfg: PPOConfig) -> np.ndarray:
    """
    构建增强观测
    
    结构: LiDAR(360) + raw_low(7) + static_feat(6) + ped_feat(5*8=40) = 413 维
    """
    obs = obs_hist[-1].astype(np.float32)
    lidar = obs[:cfg.lidar_dim]
    raw_low = obs[cfg.lidar_dim: cfg.obs_dim]
    
    # 行人预测特征 + 静态障碍物特征
    ped_features, static_features = pedestrian_predictor.update(lidar_history)
    ped_features_flat = ped_features.flatten()
    
    return np.concatenate([lidar, raw_low, static_features, ped_features_flat], axis=0).astype(np.float32)


def init_obs_history(first_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    hist: Deque[np.ndarray] = deque(maxlen=seq_len)
    for _ in range(seq_len):
        hist.append(first_obs.copy())
    return hist


def init_lidar_history(first_obs: np.ndarray, history_len: int) -> Deque[np.ndarray]:
    hist: Deque[np.ndarray] = deque(maxlen=history_len)
    for _ in range(history_len):
        hist.append(first_obs.copy())
    return hist


def init_seq_history(first_enhanced_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    hist: Deque[np.ndarray] = deque(maxlen=seq_len)
    for _ in range(seq_len):
        hist.append(first_enhanced_obs.copy())
    return hist


# =========================
# 模型
# =========================
class CNNGRUActorCriticExp7(nn.Module):
    """
    Exp7 模型: 处理 360° LiDAR + 行人预测特征
    """
    
    def __init__(self, lidar_dim: int, low_dim: int, action_dim: int, 
                 gru_hidden_dim: int = 256):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.low_dim = low_dim
        self.action_dim = action_dim

        # CNN LiDAR 编码器 - 适配 360 维输入
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, lidar_dim)
            lidar_feat_dim = self.lidar_encoder(dummy).shape[1]

        self.lidar_fc = nn.Sequential(
            nn.Linear(lidar_feat_dim, 128),
            nn.Tanh(),
        )

        # 低维状态 + 行人特征编码器
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        # GRU 前处理
        self.pre_gru = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.Tanh(),
        )

        # GRU
        self.gru = nn.GRU(input_size=256, hidden_size=gru_hidden_dim, num_layers=1, batch_first=True)

        # GRU 后处理
        self.post_gru = nn.Sequential(
            nn.Linear(gru_hidden_dim, 256),
            nn.Tanh(),
        )

        # Actor & Critic
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Linear(256, 1)

    def encode_single_frame(self, obs_frame: torch.Tensor) -> torch.Tensor:
        lidar = obs_frame[:, :self.lidar_dim]
        low_dim = obs_frame[:, self.lidar_dim: self.lidar_dim + self.low_dim]

        lidar_feat = self.lidar_fc(self.lidar_encoder(lidar.unsqueeze(1)))
        low_feat = self.low_encoder(low_dim)
        return self.pre_gru(torch.cat([lidar_feat, low_feat], dim=-1))

    def encode_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, obs_dim = obs_seq.shape
        flat = obs_seq.reshape(bsz * seq_len, obs_dim)
        frame_feat = self.encode_single_frame(flat).reshape(bsz, seq_len, -1)
        gru_out, _ = self.gru(frame_feat)
        return self.post_gru(gru_out[:, -1, :])

    def forward(self, obs_seq: torch.Tensor):
        feat = self.encode_sequence(obs_seq)
        mean = self.actor_mean(feat)
        value = self.critic(feat)
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


def evaluate_policy(env: UnityNavEnv, model: nn.Module, cfg: PPOConfig, 
                    device: torch.device, pedestrian_predictor: PedestrianPredictor,
                    num_episodes: int = 50):
    model.eval()
    returns = []
    lengths = []
    successes = []
    collisions = []
    timeouts = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs_np, info = env.reset()
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            lidar_hist = init_lidar_history(obs_np, cfg.lidar_history_len)
            enhanced_obs = build_enhanced_obs(obs_hist, lidar_hist, pedestrian_predictor, cfg)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)

            done = False
            ep_ret = 0.0
            ep_len = 0
            last_info = info

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
                    obs_hist.append(obs_np.copy())
                    lidar_hist.append(obs_np.copy())
                    next_enhanced_obs = build_enhanced_obs(obs_hist, lidar_hist, pedestrian_predictor, cfg)
                    seq_hist.append(next_enhanced_obs.copy())

            returns.append(ep_ret)
            lengths.append(ep_len)
            successes.append(float(last_info.get("success", False)))
            collisions.append(float(last_info.get("collision", False)))
            timeouts.append(float(last_info.get("timeout", False)))

    model.train()
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "timeout_rate": float(np.mean(timeouts)),
    }


def get_env_path() -> str:
    import platform
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if platform.system() == "Linux":
        # Exp7: 360° LiDAR 环境
        linux_paths = [
            "/home/dell/DRL_Navigation/Corridor_linux_360/Corridor_linux_360.x86_64",
            os.path.join(script_dir, "Corridor_linux_360/Corridor_linux_360.x86_64"),
            "./Corridor_linux_360/Corridor_linux_360.x86_64",
        ]
        for p in linux_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError("Could not find Unity environment for Linux (360 LiDAR).")
    else:
        win_paths = [
            r"D:\DRL_Navigation\Builds\Project_1.exe",
            os.path.join(script_dir, "Builds/Project_1.exe"),
        ]
        for p in win_paths:
            if os.path.exists(p):
                return p
        raise FileNotFoundError("Could not find Unity environment for Windows.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--updates", type=int, default=None)
    args = parser.parse_args()
    
    cfg = PPOConfig()
    
    if args.resume:
        cfg.resume = True
        cfg.resume_checkpoint = args.resume
    if args.updates:
        cfg.total_updates = args.updates
    
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    env_path = args.env if args.env else get_env_path()
    print(f"Using environment: {env_path}")

    # 360° LiDAR 环境配置
    env_cfg = EnvConfig(
        file_name=env_path,
        behavior_name="Navtest?team=0",
        no_graphics=args.no_graphics,
        obs_size=367,
        lidar_dim=360,
        reach_goal_radius=0.5,
        max_steps=450,
        progress_gain=3.5,
        time_penalty=-0.005,
        collision_penalty=-20.0,
        success_bonus=120.0,
        timeout_penalty=-15.0,
        near_obstacle_threshold=0.4,
        near_obstacle_penalty=-0.2,
        action_l2_penalty=-0.0005,
    )

    device = torch.device(cfg.device)
    env = UnityNavEnv(env_cfg)
    
    # 行人预测器
    pedestrian_predictor = PedestrianPredictor(
        lidar_dim=cfg.lidar_dim,
        max_pedestrians=cfg.pedestrian_max_num,
        prediction_horizon=cfg.prediction_horizon,
    )
    
    model = CNNGRUActorCriticExp7(
        cfg.lidar_dim, cfg.low_dim, cfg.action_dim, gru_hidden_dim=256
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    writer.add_text("config", str(cfg))
    writer.add_text("env_config", str(env_cfg))

    start_update = 1
    global_step = 0
    if cfg.resume and cfg.resume_checkpoint and os.path.exists(cfg.resume_checkpoint):
        print(f"Resuming from: {cfg.resume_checkpoint}")
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device)
        # 使用 strict=True 确保模型参数完整加载
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError as e:
            print(f"Warning: 模型参数不完全匹配，尝试 strict=False 加载: {e}")
            model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 1) + 1
        global_step = ckpt.get("global_step", 0)
        if "best_success_rate" in ckpt:
            cfg.best_success_rate = ckpt["best_success_rate"]

    def lr_lambda(update):
        frac = 1.0 - (update - 1) / cfg.total_updates
        return frac * 0.9 + 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if cfg.resume and start_update > 1:
        for _ in range(1, start_update):
            scheduler.step()

    obs_np, _ = env.reset()
    obs_hist = init_obs_history(obs_np, cfg.seq_len)
    lidar_hist = init_lidar_history(obs_np, cfg.lidar_history_len)
    enhanced_obs = build_enhanced_obs(obs_hist, lidar_hist, pedestrian_predictor, cfg)
    seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)

    episode_return = 0.0
    episode_len = 0
    train_success_window = deque(maxlen=50)
    train_collision_window = deque(maxlen=50)

    start_time = time.time()

    print("\n" + "="*60)
    print("EXP7 - 360° LiDAR 行人运动预测")
    print("="*60)
    print("核心改进:")
    print(f"  - LiDAR 维度: 180 -> 360 (全向感知)")
    print(f"  - LiDAR 历史帧数: {cfg.lidar_history_len}")
    print(f"  - 最多检测行人数: {cfg.pedestrian_max_num}")
    print(f"  - 每个行人特征维度: {cfg.pedestrian_feat_dim}")
    print(f"  - 预测时间范围: {cfg.prediction_horizon}s")
    print(f"  - 增强观测维度: {cfg.enhanced_obs_dim}")
    print("="*60 + "\n")

    for update in range(start_update, cfg.total_updates + 1):
        seq_obs_buf: List[torch.Tensor] = []
        action_buf: List[torch.Tensor] = []
        logprob_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        value_buf: List[torch.Tensor] = []
        rollout_rewards = []

        for step in range(cfg.rollout_steps):
            global_step += 1
            seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(seq_tensor)
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
                train_success_window.append(float(info["success"]))
                train_collision_window.append(float(info["collision"]))

                writer.add_scalar("train/episode_return", float(episode_return), global_step)
                writer.add_scalar("train/episode_length", int(episode_len), global_step)
                writer.add_scalar("train/episode_success", float(info["success"]), global_step)
                writer.add_scalar("train/episode_collision", float(info["collision"]), global_step)

                print(
                    f"[train ep] u={update:04d} s={global_step} "
                    f"ret={episode_return:.1f} len={episode_len} "
                    f"succ={info['success']} coll={info['collision']}"
                )

                next_obs_np, _ = env.reset()
                obs_hist = init_obs_history(next_obs_np, cfg.seq_len)
                lidar_hist = init_lidar_history(next_obs_np, cfg.lidar_history_len)
                enhanced_obs = build_enhanced_obs(obs_hist, lidar_hist, pedestrian_predictor, cfg)
                seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
                episode_return = 0.0
                episode_len = 0
            else:
                obs_hist.append(next_obs_np.copy())
                lidar_hist.append(next_obs_np.copy())
                next_enhanced_obs = build_enhanced_obs(obs_hist, lidar_hist, pedestrian_predictor, cfg)
                seq_hist.append(next_enhanced_obs.copy())

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

        batch_size = cfg.rollout_steps
        batch_inds = np.arange(batch_size)
        last_pg_loss, last_v_loss, last_entropy, last_kl = 0.0, 0.0, 0.0, 0.0
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
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_loss

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

        sps = int(global_step / max(time.time() - start_time, 1e-6))
        writer.add_scalar("train/update", update, global_step)
        writer.add_scalar("train/loss_pi", last_pg_loss, global_step)
        writer.add_scalar("train/loss_v", last_v_loss, global_step)
        writer.add_scalar("train/entropy", last_entropy, global_step)
        writer.add_scalar("train/approx_kl", last_kl, global_step)
        writer.add_scalar("train/SPS", sps, global_step)

        if train_success_window:
            writer.add_scalar("train_window/success_rate_50", float(np.mean(train_success_window)), global_step)
            writer.add_scalar("train_window/collision_rate_50", float(np.mean(train_collision_window)), global_step)

        print(f"update={update:04d} loss={last_pg_loss:.3f} ent={last_entropy:.3f} kl={last_kl:.4f}")

        scheduler.step()

        if update % cfg.eval_every == 0:
            eval_stats = evaluate_policy(env, model, cfg, device, pedestrian_predictor, num_episodes=cfg.eval_episodes)
            writer.add_scalar("eval/success_rate", eval_stats["success_rate"], global_step)
            writer.add_scalar("eval/collision_rate", eval_stats["collision_rate"], global_step)
            writer.add_scalar("eval/timeout_rate", eval_stats["timeout_rate"], global_step)

            print(f"[EVAL] u={update:04d} succ={eval_stats['success_rate']:.1%} "
                  f"coll={eval_stats['collision_rate']:.1%} to={eval_stats['timeout_rate']:.1%}")

            if cfg.save_best and eval_stats["success_rate"] > cfg.best_success_rate:
                cfg.best_success_rate = eval_stats["success_rate"]
                best_path = os.path.join(cfg.save_dir, "best_model.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "best_success_rate": cfg.best_success_rate,
                }, best_path)
                print(f"  -> New best: {cfg.best_success_rate:.1%}")

            obs_np, _ = env.reset()
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            lidar_hist = init_lidar_history(obs_np, cfg.lidar_history_len)
            enhanced_obs = build_enhanced_obs(obs_hist, lidar_hist, pedestrian_predictor, cfg)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
            episode_return = 0.0
            episode_len = 0

        if update % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"ppo_update_{update:04d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "best_success_rate": cfg.best_success_rate,
            }, save_path)
            print(f"Saved: {save_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()

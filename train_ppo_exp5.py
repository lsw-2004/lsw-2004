"""
实验5: 目标突破 0.6 成功率的改进训练脚本

核心改进:
1. 动态障碍物检测特征 - 区分静态/动态障碍物
2. Entropy coefficient 衰减 - 后期策略收敛
3. 辅助任务 - LiDAR预测 + 碰撞预测
4. 风险感知奖励 - 智能避障激励
5. 课程学习 - 渐进式难度提升
6. 增强的时序建模 - 更长的序列 + 更好的特征
"""
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Optional

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # 训练参数
    total_updates: int = 3000          # 更多训练轮数
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # 优化器参数
    lr: float = 3e-4
    clip_coef: float = 0.2
    
    # Entropy 衰减 (关键改进)
    ent_coef_start: float = 0.01       # 起始: 较高探索
    ent_coef_end: float = 0.001        # 结束: 低探索, 策略收敛
    
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    update_epochs: int = 6
    minibatch_size: int = 256
    target_kl: float = 0.02            # 收紧 KL 早停

    # 观测参数
    obs_dim: int = 187
    lidar_dim: int = 180
    raw_low_dim: int = 7
    
    # 新增: 动态检测特征维度
    dynamic_feat_dim: int = 24         # 动态障碍物特征
    
    # 预测特征 (增强版)
    pred_feat_dim: int = 20            # 从 12 增加到 20
    
    action_dim: int = 2
    seq_len: int = 16                  # 从 8 增加到 16, 更长时序

    # 保存路径
    save_dir: str = "./checkpoints/cnn_gru_ppo_tb/exp5"
    log_dir: str = "./runs/cnn_gru_ppo_tb/exp5"
    save_every: int = 50

    # 特征开关
    use_dynamic_features: bool = True  # 动态障碍物检测
    use_prediction_features: bool = True
    use_aux_tasks: bool = True         # 辅助任务

    # 辅助任务权重
    aux_lidar_weight: float = 0.1      # LiDAR 预测损失权重
    aux_collision_weight: float = 0.2  # 碰撞预测损失权重

    # 在线评估
    eval_every: int = 10
    eval_episodes: int = 50
    eval_deterministic: bool = True

    # 恢复训练
    resume: bool = False
    resume_checkpoint: str = ""

    @property
    def low_dim(self) -> int:
        dim = self.raw_low_dim
        if self.use_prediction_features:
            dim += self.pred_feat_dim
        if self.use_dynamic_features:
            dim += self.dynamic_feat_dim
        return dim

    @property
    def enhanced_obs_dim(self) -> int:
        return self.lidar_dim + self.low_dim


# =========================
# 动态障碍物检测 (核心改进)
# =========================
class DynamicObstacleDetector:
    """
    基于连续 LiDAR 帧检测动态障碍物
    
    原理:
    - 静态障碍物: 距离变化仅由机器人自身运动引起
    - 动态障碍物: 距离变化包含障碍物自身运动
    
    输出特征:
    - 动态区域 mask
    - 动态物体速度估计
    - 动态威胁等级
    """
    
    def __init__(self, lidar_dim: int = 180, history_len: int = 5):
        self.lidar_dim = lidar_dim
        self.history_len = history_len
        self.lidar_history: Deque[np.ndarray] = deque(maxlen=history_len)
        
        # 区域划分
        self.left_slice = slice(0, 60)
        self.front_slice = slice(60, 120)
        self.right_slice = slice(120, 180)
        
    def reset(self):
        self.lidar_history.clear()
    
    def update(self, lidar: np.ndarray, robot_velocity: float, robot_angular: float):
        """更新 LiDAR 历史"""
        self.lidar_history.append(lidar.copy())
    
    def detect(self) -> np.ndarray:
        """
        检测动态障碍物特征
        
        返回 24 维特征:
        - [0:3] 左/前/右动态区域比例
        - [3:6] 左/前/右动态物体平均速度
        - [6:9] 左/前/右最大动态速度
        - [9:12] 左/前/右动态威胁等级
        - [12:15] 动态障碍物数量估计
        - [15:18] 最近动态障碍物距离
        - [18:21] 动态障碍物方向分布
        - [21:24] 全局动态风险指标
        """
        if len(self.lidar_history) < 2:
            return np.zeros(24, dtype=np.float32)
        
        # 计算距离变化
        curr = self.lidar_history[-1]
        prev = self.lidar_history[-2]
        
        # 距离变化 = 动态物体的贡献
        # (机器人运动引起的距离变化是系统性的, 动态物体引起的是局部的)
        delta = curr - prev
        
        # 使用局部异常检测: 如果某点的变化显著大于周围点, 则可能是动态物体
        # 平滑后做比较
        delta_smooth = np.convolve(delta, np.ones(5)/5, mode='same')
        local_var = np.abs(delta - delta_smooth)
        
        # 动态区域阈值: 变化超过噪声水平
        dynamic_threshold = 0.02  # 归一化距离变化
        dynamic_mask = (np.abs(delta) > dynamic_threshold) & (local_var > 0.01)
        
        # 动态物体速度估计 (归一化, 需要乘以 10m 才是实际速度)
        dynamic_velocity = np.abs(delta) * dynamic_mask
        
        # 分区域统计
        features = np.zeros(24, dtype=np.float32)
        
        for i, region_slice in enumerate([self.left_slice, self.front_slice, self.right_slice]):
            region_mask = dynamic_mask[region_slice]
            region_vel = dynamic_velocity[region_slice]
            region_delta = delta[region_slice]
            
            # 动态区域比例
            features[i] = float(np.mean(region_mask.astype(np.float32)))
            
            # 平均动态速度
            if np.any(region_mask):
                features[3 + i] = float(np.mean(region_vel[region_mask]))
                features[6 + i] = float(np.max(region_vel))
            else:
                features[3 + i] = 0.0
                features[6 + i] = 0.0
            
            # 动态威胁等级 = 动态比例 × 速度
            curr_region = curr[region_slice]
            threat = features[i] * features[3 + i] * (1.0 - np.min(curr_region))
            features[9 + i] = float(np.clip(threat, 0, 1))
        
        # 动态障碍物数量估计 (连通域分析简化版)
        dynamic_indices = np.where(dynamic_mask)[0]
        if len(dynamic_indices) > 0:
            # 简单估计: 相邻点归为一个物体
            num_objects = 1
            for j in range(1, len(dynamic_indices)):
                if dynamic_indices[j] - dynamic_indices[j-1] > 5:  # 间隔超过5个点
                    num_objects += 1
            features[12] = float(min(num_objects, 5))  # 最多估计5个
        else:
            features[12] = 0.0
        
        # 最近动态障碍物距离
        if np.any(dynamic_mask):
            dynamic_distances = curr[dynamic_mask]
            features[15] = float(np.min(dynamic_distances))  # 最近距离
            features[16] = float(np.mean(dynamic_distances))  # 平均距离
            features[17] = float(np.std(dynamic_distances))   # 距离方差
        else:
            features[15:18] = 1.0  # 无动态障碍物时, 设为远距离
        
        # 动态障碍物方向分布
        if np.any(dynamic_mask):
            left_dynamic = np.sum(dynamic_mask[self.left_slice])
            front_dynamic = np.sum(dynamic_mask[self.front_slice])
            right_dynamic = np.sum(dynamic_mask[self.right_slice])
            total = left_dynamic + front_dynamic + right_dynamic + 1e-6
            features[18] = left_dynamic / total
            features[19] = front_dynamic / total
            features[20] = right_dynamic / total
        else:
            features[18:21] = 0.0
        
        # 全局动态风险指标
        features[21] = float(np.mean(dynamic_mask.astype(np.float32)))  # 整体动态比例
        features[22] = float(np.mean(np.abs(delta)))  # 整体变化强度
        features[23] = float(np.max(dynamic_velocity))  # 最大动态速度
        
        return features


# =========================
# 增强的预测特征
# =========================
def _safe_stats(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    return float(np.min(x)), float(np.mean(x))


def build_enhanced_prediction_features(
    obs_hist: Deque[np.ndarray], 
    lidar_dim: int,
    dynamic_detector: Optional[DynamicObstacleDetector] = None
) -> np.ndarray:
    """
    构建增强的预测特征 (20维)
    
    包括:
    - 原有的12维基础特征
    - 新增8维趋势特征
    """
    if len(obs_hist) == 0:
        return np.zeros(20, dtype=np.float32)

    curr_obs = obs_hist[-1]
    curr_lidar = curr_obs[:lidar_dim].astype(np.float32)
    
    # 获取历史帧
    prev_lidar = curr_lidar.copy() if len(obs_hist) == 1 else obs_hist[-2][:lidar_dim].astype(np.float32)
    
    # 三帧历史 (如果有的话)
    prev2_lidar = prev_lidar.copy() if len(obs_hist) < 3 else obs_hist[-3][:lidar_dim].astype(np.float32)

    # 区域划分
    left = slice(0, 60)
    front = slice(60, 120)
    right = slice(120, 180)

    def get_region_stats(lidar, region):
        data = lidar[region]
        return float(np.min(data)), float(np.mean(data)), float(np.std(data))

    # 当前帧统计
    curr_stats = {
        'left': get_region_stats(curr_lidar, left),
        'front': get_region_stats(curr_lidar, front),
        'right': get_region_stats(curr_lidar, right),
    }
    
    prev_stats = {
        'left': get_region_stats(prev_lidar, left),
        'front': get_region_stats(prev_lidar, front),
        'right': get_region_stats(prev_lidar, right),
    }

    # 构建特征
    feat = np.zeros(20, dtype=np.float32)
    
    # [0:2] 左侧: min, mean
    feat[0] = curr_stats['left'][0]
    feat[1] = curr_stats['left'][1]
    
    # [2:4] 前方: min, mean
    feat[2] = curr_stats['front'][0]
    feat[3] = curr_stats['front'][1]
    
    # [4:6] 右侧: min, mean
    feat[4] = curr_stats['right'][0]
    feat[5] = curr_stats['right'][1]
    
    # [6:9] 接近速度 (变化率)
    feat[6] = float(prev_stats['left'][1] - curr_stats['left'][1])
    feat[7] = float(prev_stats['front'][1] - curr_stats['front'][1])
    feat[8] = float(prev_stats['right'][1] - curr_stats['right'][1])
    
    # [9:12] 接近比例 (多少点在变近)
    th = 0.02
    feat[9] = float(np.mean((prev_lidar[left] - curr_lidar[left]) > th))
    feat[10] = float(np.mean((prev_lidar[front] - curr_lidar[front]) > th))
    feat[11] = float(np.mean((prev_lidar[right] - curr_lidar[right]) > th))
    
    # [12:14] 全局特征
    feat[12] = float(np.min(prev_lidar) - np.min(curr_lidar))  # 全局最近点变化
    feat[13] = float(curr_stats['left'][1] - curr_stats['right'][1])  # 左右平衡
    
    # [14:16] 前方风险
    front_risk = feat[10] * (1.0 - curr_stats['front'][0])
    feat[14] = float(front_risk)
    feat[15] = float(curr_stats['front'][2])  # 前方距离方差
    
    # [16:20] 趋势特征 (新增)
    if len(obs_hist) >= 3:
        # 二阶变化 (加速度)
        delta1 = prev_lidar - prev2_lidar  # t-2 -> t-1
        delta2 = curr_lidar - prev_lidar   # t-1 -> t
        acceleration = delta2 - delta1
        
        feat[16] = float(np.mean(acceleration[front]))  # 前方加速度
        feat[17] = float(np.mean(np.abs(acceleration)))  # 全局加速度强度
        
        # 趋势预测
        feat[18] = float(np.mean(delta2[front] * 2))  # 前方趋势外推
        feat[19] = float(np.std(delta2))  # 变化不稳定性
    else:
        feat[16:20] = 0.0
    
    return feat


def build_enhanced_obs(
    obs_hist: Deque[np.ndarray], 
    cfg: PPOConfig,
    dynamic_detector: Optional[DynamicObstacleDetector] = None
) -> np.ndarray:
    """构建增强的观测向量"""
    obs = obs_hist[-1].astype(np.float32)
    lidar = obs[:cfg.lidar_dim]
    low = obs[cfg.lidar_dim: cfg.obs_dim]
    
    # 预测特征
    if cfg.use_prediction_features:
        pred_feat = build_enhanced_prediction_features(obs_hist, cfg.lidar_dim, dynamic_detector)
        low = np.concatenate([low, pred_feat], axis=0)
    
    # 动态障碍物特征
    if cfg.use_dynamic_features and dynamic_detector is not None:
        dynamic_feat = dynamic_detector.detect()
        low = np.concatenate([low, dynamic_feat], axis=0)
    
    return np.concatenate([lidar, low], axis=0).astype(np.float32)


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


# =========================
# 带辅助任务的模型
# =========================
class CNNGRUActorCriticWithAux(nn.Module):
    """
    带辅助任务的 Actor-Critic 模型
    
    辅助任务:
    1. LiDAR 预测: 预测下一帧 LiDAR
    2. 碰撞预测: 预测未来碰撞概率
    """
    
    def __init__(
        self, 
        lidar_dim: int, 
        low_dim: int, 
        action_dim: int, 
        gru_hidden_dim: int = 256,
        use_aux_tasks: bool = True
    ):
        super().__init__()
        self.lidar_dim = lidar_dim
        self.low_dim = low_dim
        self.action_dim = action_dim
        self.use_aux_tasks = use_aux_tasks

        # LiDAR 编码器 (增强版)
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),  # 增加通道数
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # 增加通道数
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, lidar_dim)
            lidar_feat_dim = self.lidar_encoder(dummy).shape[1]

        self.lidar_fc = nn.Sequential(
            nn.Linear(lidar_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        # 低维状态编码器
        self.low_encoder = nn.Sequential(
            nn.Linear(low_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        # GRU 前处理
        self.pre_gru = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
        )

        # GRU (2层, 增强时序建模)
        self.gru = nn.GRU(
            input_size=256, 
            hidden_size=gru_hidden_dim, 
            num_layers=2,  # 增加到2层
            batch_first=True,
            dropout=0.1
        )

        # GRU 后处理
        self.post_gru = nn.Sequential(
            nn.Linear(gru_hidden_dim, 256),
            nn.ReLU(),
        )

        # Actor
        self.actor_mean = nn.Linear(256, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic
        self.critic = nn.Linear(256, 1)
        
        # 辅助任务头
        if self.use_aux_tasks:
            # LiDAR 预测: 预测压缩后的 LiDAR 特征 (而非原始180维)
            self.lidar_predictor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),  # 预测 LiDAR 的压缩特征
            )
            
            # 碰撞预测: 预测未来 1/3/5/10 步的碰撞概率
            self.collision_predictor = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 4),  # 4个时间尺度
                nn.Sigmoid(),
            )

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
        
        if self.use_aux_tasks:
            pred_lidar_feat = self.lidar_predictor(feat)
            pred_collision = self.collision_predictor(feat)
            return mean, logstd, value, pred_lidar_feat, pred_collision
        else:
            return mean, logstd, value, None, None

    def get_action_and_value(self, obs_seq: torch.Tensor, action: torch.Tensor = None):
        outputs = self.forward(obs_seq)
        mean, logstd, value = outputs[0], outputs[1], outputs[2]
        
        std = torch.exp(logstd)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        
        if self.use_aux_tasks:
            return action, logprob, entropy, value.squeeze(-1), outputs[3], outputs[4]
        else:
            return action, logprob, entropy, value.squeeze(-1), None, None

    def get_value(self, obs_seq: torch.Tensor):
        feat = self.encode_sequence(obs_seq)
        return self.critic(feat).squeeze(-1)

    def get_deterministic_action(self, obs_seq: torch.Tensor):
        feat = self.encode_sequence(obs_seq)
        mean = self.actor_mean(feat)
        value = self.critic(feat)
        return mean, value.squeeze(-1)


# =========================
# GAE 计算
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


# =========================
# 课程学习奖励
# =========================
def compute_curriculum_reward(
    base_reward: float,
    info: dict,
    obs: np.ndarray,
    update: int,
    total_updates: int,
    cfg: PPOConfig
) -> float:
    """
    课程学习奖励:
    - 早期: 宽松惩罚, 鼓励探索
    - 中期: 逐步收紧
    - 后期: 严格惩罚碰撞, 鼓励效率
    """
    progress = update / total_updates
    
    # 课程阶段
    if progress < 0.3:
        # 早期: 探索阶段
        collision_multiplier = 0.5
        time_multiplier = 0.5
        risk_penalty_weight = 0.1
    elif progress < 0.7:
        # 中期: 学习阶段
        collision_multiplier = 1.0
        time_multiplier = 1.0
        risk_penalty_weight = 0.3
    else:
        # 后期: 收敛阶段
        collision_multiplier = 1.5
        time_multiplier = 1.5
        risk_penalty_weight = 0.5
    
    reward = base_reward
    
    # 风险感知惩罚
    lidar = obs[:180] * 10.0  # 转为米
    min_dist = np.min(lidar)
    cur_v = float(obs[184])
    
    # 高速 + 近距离 = 危险
    if min_dist < 1.0:
        danger_level = (1.0 - min_dist) * max(0, cur_v)
        reward -= risk_penalty_weight * danger_level
    
    # 安全通行奖励: 成功穿过狭窄区域
    if min_dist < 0.5 and not info.get('collision', False):
        reward += 0.2 * (1.0 - progress)  # 早期鼓励探索狭窄通道
    
    return reward


# =========================
# 评估函数
# =========================
def evaluate_policy(
    env: UnityNavEnv, 
    model: CNNGRUActorCriticWithAux, 
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
            
            # 初始化观测历史和动态检测器
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            dynamic_detector = DynamicObstacleDetector(cfg.lidar_dim, history_len=5) if cfg.use_dynamic_features else None
            
            enhanced_obs = build_enhanced_obs(obs_hist, cfg, dynamic_detector)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)

            done = False
            ep_ret = 0.0
            ep_len = 0
            last_info = info
            
            if dynamic_detector:
                dynamic_detector.reset()

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
                    
                    # 更新动态检测器
                    if dynamic_detector:
                        lidar = obs_np[:cfg.lidar_dim] * 10.0
                        cur_v = float(obs_np[184])
                        cur_w = float(obs_np[185])
                        dynamic_detector.update(obs_np[:cfg.lidar_dim], cur_v, cur_w)
                    
                    next_enhanced_obs = build_enhanced_obs(obs_hist, cfg, dynamic_detector)
                    seq_hist.append(next_enhanced_obs.copy())

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
# =========================
def main():
    cfg = PPOConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 环境配置
    env_cfg = EnvConfig(
        file_name=r"D:\DRL_Navigation\Builds\Project_1.exe",
        behavior_name="Navtest?team=0",
        no_graphics=False,
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=450,
        progress_gain=3.0,               # 进一步提高
        time_penalty=-0.008,             # 更强的时间压力
        collision_penalty=-10.0,         # 更严厉的碰撞惩罚
        success_bonus=100.0,             # 更高的成功奖励
        timeout_penalty=-20.0,           # 更严厉的超时惩罚
        near_obstacle_threshold=0.5,
        near_obstacle_penalty=-0.2,
        action_l2_penalty=-0.001,
    )

    device = torch.device(cfg.device)
    env = UnityNavEnv(env_cfg)
    
    # 创建模型
    model = CNNGRUActorCriticWithAux(
        cfg.lidar_dim, 
        cfg.low_dim, 
        cfg.action_dim, 
        gru_hidden_dim=256,
        use_aux_tasks=cfg.use_aux_tasks
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    writer.add_text("config", str(cfg))

    # 恢复训练
    start_update = 1
    global_step = 0
    if cfg.resume and cfg.resume_checkpoint:
        if os.path.exists(cfg.resume_checkpoint):
            print(f"Resuming from checkpoint: {cfg.resume_checkpoint}")
            ckpt = torch.load(cfg.resume_checkpoint, map_location=device)
            model.load_state_dict(ckpt["model"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_update = ckpt.get("update", 1) + 1
            global_step = ckpt.get("global_step", 0)
            print(f"Resumed from update {start_update - 1}, global_step {global_step}")
        else:
            print(f"Checkpoint not found: {cfg.resume_checkpoint}, starting from scratch")

    # 学习率调度器
    def lr_lambda(update):
        frac = 1.0 - (update - 1) / cfg.total_updates
        return frac * 0.9 + 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    if cfg.resume and cfg.resume_checkpoint and start_update > 1:
        for _ in range(1, start_update):
            scheduler.step()

    # 初始化
    obs_np, _ = env.reset()
    obs_hist = init_obs_history(obs_np, cfg.seq_len)
    dynamic_detector = DynamicObstacleDetector(cfg.lidar_dim, history_len=5) if cfg.use_dynamic_features else None
    enhanced_obs = build_enhanced_obs(obs_hist, cfg, dynamic_detector)
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

    # =========================
    # 主训练循环
    # =========================
    for update in range(start_update, cfg.total_updates + 1):
        seq_obs_buf: List[torch.Tensor] = []
        action_buf: List[torch.Tensor] = []
        logprob_buf: List[torch.Tensor] = []
        reward_buf: List[torch.Tensor] = []
        done_buf: List[torch.Tensor] = []
        value_buf: List[torch.Tensor] = []
        
        # 辅助任务数据
        lidar_feat_buf: List[torch.Tensor] = []
        collision_label_buf: List[torch.Tensor] = []

        rollout_rewards = []

        # Rollout
        for step in range(cfg.rollout_steps):
            global_step += 1
            seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, logprob, entropy, value, pred_lidar_feat, pred_collision = model.get_action_and_value(seq_tensor)
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
                value = value.squeeze(0)

            action_np = action.detach().cpu().numpy()
            next_obs_np, base_reward, done, truncated, info = env.step(action_np)
            
            # 课程学习奖励
            reward = compute_curriculum_reward(
                base_reward, info, next_obs_np, update, cfg.total_updates, cfg
            )
            
            episode_return += reward
            episode_len += 1
            rollout_rewards.append(reward)

            seq_obs_buf.append(seq_tensor.squeeze(0).detach())
            action_buf.append(action.detach())
            logprob_buf.append(logprob.detach())
            reward_buf.append(torch.tensor(reward, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))
            value_buf.append(value.detach())
            
            # 保存辅助任务数据
            if cfg.use_aux_tasks:
                # 下一帧 LiDAR 特征 (简化: 使用压缩特征)
                next_lidar = next_obs_np[:cfg.lidar_dim]
                lidar_feat_buf.append(torch.tensor(next_lidar, dtype=torch.float32, device=device))
                
                # 碰撞标签 (简化: 当前是否接近碰撞)
                is_dangerous = float(np.min(next_lidar) < 0.1)
                collision_label_buf.append(torch.tensor(is_dangerous, dtype=torch.float32, device=device))

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
                    f"success={info['success']} collision={info['collision']} timeout={info['timeout']}"
                )

                next_obs_np, _ = env.reset()
                obs_hist = init_obs_history(next_obs_np, cfg.seq_len)
                if dynamic_detector:
                    dynamic_detector.reset()
                enhanced_obs = build_enhanced_obs(obs_hist, cfg, dynamic_detector)
                seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
                episode_return = 0.0
                episode_len = 0
            else:
                obs_hist.append(next_obs_np.copy())
                if dynamic_detector:
                    cur_v = float(next_obs_np[184])
                    cur_w = float(next_obs_np[185])
                    dynamic_detector.update(next_obs_np[:cfg.lidar_dim], cur_v, cur_w)
                next_enhanced_obs = build_enhanced_obs(obs_hist, cfg, dynamic_detector)
                seq_hist.append(next_enhanced_obs.copy())

        # 计算 GAE
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
            rewards=reward_buf,
            dones=done_buf,
            values=value_buf,
            next_value=next_value,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算 entropy coefficient (衰减)
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

                _, newlogprob, entropy, newvalue, pred_lidar, pred_collision = model.get_action_and_value(
                    mb_seq_obs, mb_actions
                )

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
                if cfg.use_aux_tasks and pred_lidar is not None:
                    mb_lidar_feat = torch.stack([lidar_feat_buf[i] for i in mb_inds])
                    mb_collision_label = torch.stack([collision_label_buf[i] for i in mb_inds])
                    
                    # LiDAR 预测损失 (简化: 直接预测原始 LiDAR)
                    lidar_loss = F.mse_loss(pred_lidar, mb_lidar_feat[:, :64])  # 只预测前64维
                    
                    # 碰撞预测损失
                    collision_loss = F.binary_cross_entropy(
                        pred_collision[:, 0],  # 预测最近时间步
                        mb_collision_label
                    )
                    
                    aux_loss = cfg.aux_lidar_weight * lidar_loss + cfg.aux_collision_weight * collision_loss
                    loss += aux_loss
                    last_aux_loss = float(aux_loss.item())

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
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
        writer.add_scalar("train/global_step", global_step, global_step)
        writer.add_scalar("train/rollout_reward_mean", float(np.mean(rollout_rewards)), global_step)
        writer.add_scalar("train/loss_pi", last_pg_loss, global_step)
        writer.add_scalar("train/loss_v", last_v_loss, global_step)
        writer.add_scalar("train/entropy", last_entropy, global_step)
        writer.add_scalar("train/approx_kl", last_kl, global_step)
        writer.add_scalar("train/clipfrac", float(np.mean(clipfracs)) if clipfracs else 0.0, global_step)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("train/ent_coef", current_ent_coef, global_step)
        writer.add_scalar("train/SPS", sps, global_step)
        writer.add_scalar("train/episodes_seen", train_ep_count, global_step)
        writer.add_scalar("train/kl_early_stop", float(early_stop), global_step)
        if cfg.use_aux_tasks:
            writer.add_scalar("train/aux_loss", last_aux_loss, global_step)

        if train_returns_window:
            writer.add_scalar("train_window/return_mean_50", float(np.mean(train_returns_window)), global_step)
            writer.add_scalar("train_window/length_mean_50", float(np.mean(train_lengths_window)), global_step)
            writer.add_scalar("train_window/success_rate_50", float(np.mean(train_success_window)), global_step)
            writer.add_scalar("train_window/collision_rate_50", float(np.mean(train_collision_window)), global_step)
            writer.add_scalar("train_window/timeout_rate_50", float(np.mean(train_timeout_window)), global_step)

        print(
            f"update={update:04d} loss_pi={last_pg_loss:.4f} loss_v={last_v_loss:.4f} "
            f"entropy={last_entropy:.4f} kl={last_kl:.5f} ent_coef={current_ent_coef:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} sps={sps}"
        )

        scheduler.step()

        # 评估
        if update % cfg.eval_every == 0:
            eval_stats = evaluate_policy(env, model, cfg, device, num_episodes=cfg.eval_episodes)
            writer.add_scalar("eval/return_mean", eval_stats["return_mean"], global_step)
            writer.add_scalar("eval/return_std", eval_stats["return_std"], global_step)
            writer.add_scalar("eval/length_mean", eval_stats["length_mean"], global_step)
            writer.add_scalar("eval/success_rate", eval_stats["success_rate"], global_step)
            writer.add_scalar("eval/collision_rate", eval_stats["collision_rate"], global_step)
            writer.add_scalar("eval/timeout_rate", eval_stats["timeout_rate"], global_step)
            writer.add_scalar("eval/final_goal_dist_mean", eval_stats["final_goal_dist_mean"], global_step)

            print(
                f"[eval] update={update:04d} ret={eval_stats['return_mean']:.3f}±{eval_stats['return_std']:.3f} "
                f"succ={eval_stats['success_rate']:.3f} coll={eval_stats['collision_rate']:.3f} "
                f"timeout={eval_stats['timeout_rate']:.3f} len={eval_stats['length_mean']:.1f}"
            )

            # 重置训练状态
            obs_np, _ = env.reset()
            obs_hist = init_obs_history(obs_np, cfg.seq_len)
            if dynamic_detector:
                dynamic_detector.reset()
            enhanced_obs = build_enhanced_obs(obs_hist, cfg, dynamic_detector)
            seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
            episode_return = 0.0
            episode_len = 0

        # 保存 checkpoint
        if update % cfg.save_every == 0:
            save_path = os.path.join(cfg.save_dir, f"ppo_gru_update_{update:04d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "env_cfg": env_cfg.__dict__,
                    "ppo_cfg": cfg.__dict__,
                    "model_type": "cnn_gru_ppo_exp5",
                },
                save_path,
            )
            print(f"saved to {save_path}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()

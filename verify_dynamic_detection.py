"""
动态检测验证脚本

用于验证 UniformMotionDetector 是否能正确检测匀速直线运动的行人。

功能:
1. 实时可视化 LiDAR 和检测结果
2. 输出检测到的动态物体信息
3. 验证速度估计的准确性
4. 统计检测成功率

使用方法:
python verify_dynamic_detection.py --episodes 10 --visualize
"""

import argparse
import time
import math
from collections import deque
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from matplotlib.collections import LineCollection

from unity_env import UnityNavEnv, EnvConfig


# =========================
# 动态检测器 (从 train_ppo_exp5_v2.py 复制)
# =========================
class UniformMotionDetector:
    """
    专门针对匀速直线运动行人的检测器
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
        self.lidar_history: deque = deque(maxlen=history_len)
        self.robot_pose_history: deque = deque(maxlen=history_len)
        
        # LiDAR 角度
        self.angles = np.linspace(-np.pi, np.pi, lidar_dim, endpoint=False)
        
        # 缓存上一次检测结果
        self.last_detected_objects: List[Dict] = []
        
    def reset(self):
        self.lidar_history.clear()
        self.robot_pose_history.clear()
        self.last_detected_objects = []
    
    def update(
        self, 
        lidar: np.ndarray, 
        robot_x: float = 0.0, 
        robot_z: float = 0.0, 
        robot_yaw: float = 0.0
    ):
        self.lidar_history.append(lidar.copy())
        self.robot_pose_history.append((robot_x, robot_z, robot_yaw))
    
    def detect(self) -> np.ndarray:
        """检测动态物体并返回特征向量"""
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
        if len(self.lidar_history) < 3:
            return np.zeros((self.lidar_dim, 2))
        
        n = len(self.lidar_history)
        times = np.arange(n) * 0.1
        
        velocities = np.zeros((self.lidar_dim, 2))
        
        for i in range(self.lidar_dim):
            distances = np.array([hist[i] for hist in self.lidar_history])
            distances_m = distances * 10.0
            
            t_mean = np.mean(times)
            d_mean = np.mean(distances_m)
            
            cov = np.sum((times - t_mean) * (distances_m - d_mean))
            var = np.sum((times - t_mean) ** 2)
            
            if var > 1e-6:
                radial_velocity = cov / var
            else:
                radial_velocity = 0.0
            
            velocity_magnitude = np.std(distances_m) / (times[-1] - times[0] + 0.1)
            tangential_velocity = np.sqrt(max(0, velocity_magnitude**2 - radial_velocity**2))
            
            velocities[i] = [radial_velocity, tangential_velocity]
        
        return velocities
    
    def _cluster_dynamic_points(self, velocities: np.ndarray) -> List[Dict]:
        curr_lidar = self.lidar_history[-1]
        
        speed_threshold = 0.2
        speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        dynamic_mask = speeds > speed_threshold
        
        distance_threshold = 8.0
        distance_mask = curr_lidar * 10.0 < distance_threshold
        
        valid_mask = dynamic_mask & distance_mask
        
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
                        'velocities': velocities[cluster_start:i],
                        'distances': curr_lidar[cluster_start:i]
                    })
        
        if in_cluster and self.lidar_dim - cluster_start >= 3:
            clusters.append({
                'indices': list(range(cluster_start, self.lidar_dim)),
                'velocities': velocities[cluster_start:self.lidar_dim],
                'distances': curr_lidar[cluster_start:self.lidar_dim]
            })
        
        return clusters
    
    def _estimate_object_properties(self, clusters: List[Dict]) -> List[Dict]:
        objects = []
        
        # 行人最大合理速度 (m/s)
        MAX_PEDESTRIAN_SPEED = 3.0
        
        for cluster in clusters[:self.max_objects]:
            indices = cluster['indices']
            vels = cluster['velocities']
            dists = cluster['distances']
            
            center_angle = np.mean(self.angles[indices])
            min_dist_idx = np.argmin(dists)
            distance = float(dists[min_dist_idx] * 10.0)
            
            mean_radial_vel = float(np.mean(vels[:, 0]))
            mean_tangential_vel = float(np.mean(vels[:, 1]))
            
            # 速度上限过滤 (行人最大约 3 m/s，超过则认为是噪声)
            total_speed = np.sqrt(mean_radial_vel**2 + mean_tangential_vel**2)
            if total_speed > MAX_PEDESTRIAN_SPEED:
                scale = MAX_PEDESTRIAN_SPEED / total_speed
                mean_radial_vel *= scale
                mean_tangential_vel *= scale
            
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
        for obj in objects:
            distance = obj['distance']
            radial_vel = obj['radial_velocity']
            tangential_vel = obj['tangential_velocity']
            angle = obj['angle']
            
            approach_factor = max(0, -radial_vel) / 2.0
            distance_factor = 1.0 - min(distance / 5.0, 1.0)
            angle_factor = 1.0 - abs(angle) / np.pi
            
            threat = approach_factor * distance_factor * angle_factor
            obj['threat'] = float(np.clip(threat, 0, 1))
            
            if radial_vel < -0.1:
                ttc = distance / (-radial_vel)
                obj['ttc'] = float(min(ttc, 10.0))
            else:
                obj['ttc'] = 10.0
            
            future_distance = distance + radial_vel * 1.0
            future_angle = angle + np.arcsin(np.clip(tangential_vel / (distance + 0.1) * 1.0, -1, 1))
            obj['future_pos_1s'] = (future_distance, future_angle)
            
            if abs(tangential_vel) > 0.3:
                obj['crossable'] = float(min(abs(tangential_vel) / 1.0, 1.0))
            else:
                obj['crossable'] = 0.0
        
        return objects
    
    def _build_features(self, objects: List[Dict]) -> np.ndarray:
        features = np.zeros(48, dtype=np.float32)
        
        n_objects = min(len(objects), self.max_objects)
        
        for i in range(n_objects):
            obj = objects[i]
            features[i] = obj['distance'] / 10.0
            features[5 + i] = obj['angle'] / np.pi
            features[10 + i] = obj['radial_velocity'] / 2.0
            features[15 + i] = obj['tangential_velocity'] / 2.0
            features[20 + i] = obj['threat']
            features[25 + i] = obj.get('crossable', 0)
            features[30 + i] = obj['ttc'] / 10.0
            features[35 + i] = obj['future_pos_1s'][0] / 10.0
        
        if n_objects > 0:
            features[40] = n_objects / self.max_objects
            features[41] = min(obj['distance'] for obj in objects) / 10.0
            features[42] = max(obj['threat'] for obj in objects)
            features[43] = min(obj['ttc'] for obj in objects) / 10.0
            features[44] = sum(1 for obj in objects if obj['radial_velocity'] < 0) / max(n_objects, 1)
            features[45] = sum(obj['threat'] for obj in objects) / max(n_objects, 1)
            features[46] = np.mean([obj['angle'] for obj in objects]) / np.pi
            features[47] = np.std([obj['angle'] for obj in objects]) / np.pi
        
        return features
    
    def get_detected_objects(self) -> List[Dict]:
        """获取检测到的物体列表"""
        return self.last_detected_objects


# =========================
# 可视化
# =========================
class DetectionVisualizer:
    """动态检测结果可视化"""
    
    def __init__(self, lidar_dim: int = 180):
        self.lidar_dim = lidar_dim
        self.angles = np.linspace(-np.pi, np.pi, lidar_dim, endpoint=False)
        
        # 创建图形
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 子图1: LiDAR 极坐标视图
        self.ax_lidar = self.fig.add_subplot(2, 2, 1, projection='polar')
        self.ax_lidar.set_title('LiDAR View', fontsize=12)
        self.ax_lidar.set_ylim(0, 10)
        
        # 子图2: 鸟瞰图
        self.ax_bev = self.axes[0, 1]
        self.ax_bev.set_title('Bird Eye View', fontsize=12)
        self.ax_bev.set_xlim(-8, 8)
        self.ax_bev.set_ylim(-8, 8)
        self.ax_bev.set_aspect('equal')
        self.ax_bev.grid(True, alpha=0.3)
        
        # 子图3: 速度分布
        self.ax_vel = self.axes[1, 0]
        self.ax_vel.set_title('Velocity Distribution', fontsize=12)
        self.ax_vel.set_xlabel('Angle (deg)')
        self.ax_vel.set_ylabel('Velocity (m/s)')
        
        # 子图4: 检测信息
        self.ax_info = self.axes[1, 1]
        self.ax_info.set_title('Detection Info', fontsize=12)
        self.ax_info.axis('off')
        
        plt.tight_layout()
        
        # 历史轨迹
        self.trajectory_x = []
        self.trajectory_y = []
        self.object_history = {}  # 追踪检测到的物体
    
    def update(
        self, 
        lidar: np.ndarray, 
        detected_objects: List[Dict],
        robot_x: float = 0.0,
        robot_z: float = 0.0,
        robot_yaw: float = 0.0,
        goal_direction: Tuple[float, float] = (0, 1)
    ):
        """更新可视化"""
        # 清除之前的绘图
        self.ax_lidar.clear()
        self.ax_bev.clear()
        self.ax_vel.clear()
        self.ax_info.clear()
        
        # === 子图1: LiDAR 极坐标 ===
        self.ax_lidar.set_title('LiDAR View (Blue=Static, Red=Dynamic)', fontsize=10)
        lidar_m = lidar * 10.0
        
        # 标记动态区域
        dynamic_angles = []
        static_angles = []
        dynamic_distances = []
        static_distances = []
        
        dynamic_indices = set()
        for obj in detected_objects:
            # 找到该物体的 LiDAR 索引范围
            angle = obj['angle']
            angular_size = obj['angular_size']
            for i, a in enumerate(self.angles):
                if abs(self._normalize_angle(a - angle)) < angular_size / 2:
                    dynamic_indices.add(i)
        
        for i, (a, d) in enumerate(zip(self.angles, lidar_m)):
            if i in dynamic_indices:
                dynamic_angles.append(a)
                dynamic_distances.append(d)
            else:
                static_angles.append(a)
                static_distances.append(d)
        
        # 绘制静态点
        if static_angles:
            self.ax_lidar.scatter(static_angles, static_distances, c='blue', s=5, alpha=0.5, label='Static')
        
        # 绘制动态点
        if dynamic_angles:
            self.ax_lidar.scatter(dynamic_angles, dynamic_distances, c='red', s=15, alpha=0.8, label='Dynamic')
        
        # 绘制检测到的物体
        for obj in detected_objects:
            angle = obj['angle']
            distance = obj['distance']
            threat = obj['threat']
            
            # 颜色表示威胁等级
            color = plt.cm.RdYlGn_r(threat)
            self.ax_lidar.scatter(angle, distance, c=[color], s=100, marker='o', edgecolors='black', linewidths=2)
            
            # 标注速度
            v_r = obj['radial_velocity']
            v_t = obj['tangential_velocity']
            self.ax_lidar.annotate(
                f'v={np.sqrt(v_r**2+v_t**2):.1f}m/s',
                (angle, distance),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )
        
        self.ax_lidar.set_ylim(0, 10)
        self.ax_lidar.legend(loc='upper right', fontsize=8)
        
        # === 子图2: 鸟瞰图 ===
        self.ax_bev.set_title('Bird Eye View', fontsize=10)
        self.ax_bev.set_xlim(-8, 8)
        self.ax_bev.set_ylim(-8, 8)
        self.ax_bev.set_aspect('equal')
        self.ax_bev.grid(True, alpha=0.3)
        
        # 绘制机器人
        robot_circle = Circle((robot_x, robot_z), 0.3, color='green', fill=True)
        self.ax_bev.add_patch(robot_circle)
        
        # 绘制机器人朝向
        dx = 0.8 * np.cos(robot_yaw)
        dy = 0.8 * np.sin(robot_yaw)
        self.ax_bev.arrow(robot_x, robot_z, dx, dy, head_width=0.2, color='green')
        
        # 绘制目标方向
        goal_x, goal_y = goal_direction
        self.ax_bev.arrow(robot_x, robot_z, goal_x * 0.5, goal_y * 0.5, head_width=0.15, color='yellow')
        
        # 绘制 LiDAR 点云
        for i, (angle, dist) in enumerate(zip(self.angles, lidar_m)):
            if dist < 9.9:
                x = robot_x + dist * np.cos(angle + robot_yaw)
                y = robot_z + dist * np.sin(angle + robot_yaw)
                if i in dynamic_indices:
                    self.ax_bev.scatter(x, y, c='red', s=10, alpha=0.5)
                else:
                    self.ax_bev.scatter(x, y, c='blue', s=3, alpha=0.3)
        
        # 绘制检测到的物体
        for i, obj in enumerate(detected_objects):
            angle = obj['angle'] + robot_yaw
            distance = obj['distance']
            x = robot_x + distance * np.cos(angle)
            y = robot_z + distance * np.sin(angle)
            
            threat = obj['threat']
            color = plt.cm.RdYlGn_r(threat)
            
            # 物体位置
            self.ax_bev.scatter(x, y, c=[color], s=150, marker='o', edgecolors='black', linewidths=2)
            
            # 速度矢量
            v_r = obj['radial_velocity']
            v_t = obj['tangential_velocity']
            # 转换到世界坐标系
            vx_world = v_r * np.cos(angle) - v_t * np.sin(angle)
            vy_world = v_r * np.sin(angle) + v_t * np.cos(angle)
            
            self.ax_bev.arrow(x, y, vx_world * 0.5, vy_world * 0.5, 
                            head_width=0.15, color='orange', linewidth=2)
            
            # 标注
            self.ax_bev.annotate(f'#{i+1}', (x, y), textcoords="offset points", 
                               xytext=(5, 5), fontsize=10, fontweight='bold')
            
            # 预测位置 (1秒后)
            future_dist, future_angle = obj['future_pos_1s']
            future_x = robot_x + future_dist * np.cos(future_angle + robot_yaw)
            future_y = robot_z + future_dist * np.sin(future_angle + robot_yaw)
            self.ax_bev.scatter(future_x, future_y, c='purple', s=50, marker='x', linewidths=2)
            self.ax_bev.plot([x, future_x], [y, future_y], 'purple', linestyle='--', alpha=0.5)
        
        # === 子图3: 速度分布 ===
        self.ax_vel.set_title('Radial Velocity by Angle', fontsize=10)
        
        # 计算每个点的速度 (需要历史数据)
        velocities = []
        for obj in detected_objects:
            angle_deg = np.degrees(obj['angle'])
            v_r = obj['radial_velocity']
            v_t = obj['tangential_velocity']
            speed = np.sqrt(v_r**2 + v_t**2)
            velocities.append((angle_deg, v_r, speed))
        
        if velocities:
            angles_deg = [v[0] for v in velocities]
            v_rs = [v[1] for v in velocities]
            speeds = [v[2] for v in velocities]
            
            colors = ['red' if v < 0 else 'green' for v in v_rs]  # 负=接近, 正=远离
            self.ax_vel.bar(angles_deg, speeds, color=colors, alpha=0.7, width=15)
            self.ax_vel.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            self.ax_vel.set_xlabel('Angle (deg)')
            self.ax_vel.set_ylabel('Speed (m/s)')
        
        # === 子图4: 检测信息 ===
        self.ax_info.axis('off')
        info_text = "Detected Objects:\n" + "="*30 + "\n"
        
        if detected_objects:
            for i, obj in enumerate(detected_objects):
                info_text += f"\nObject #{i+1}:\n"
                info_text += f"  Distance: {obj['distance']:.2f} m\n"
                info_text += f"  Angle: {np.degrees(obj['angle']):.1f}°\n"
                info_text += f"  Radial Vel: {obj['radial_velocity']:.2f} m/s "
                info_text += f"({'approaching' if obj['radial_velocity'] < 0 else 'leaving'})\n"
                info_text += f"  Tangential Vel: {obj['tangential_velocity']:.2f} m/s\n"
                info_text += f"  Threat: {obj['threat']:.2f}\n"
                info_text += f"  TTC: {obj['ttc']:.1f} s\n"
        else:
            info_text += "\nNo dynamic objects detected."
        
        self.ax_info.text(0.1, 0.9, info_text, transform=self.ax_info.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def _normalize_angle(self, angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# =========================
# 主验证函数
# =========================
def verify_detection(
    env_path: str,
    num_episodes: int = 5,
    steps_per_episode: int = 200,
    visualize: bool = True,
    verbose: bool = True
):
    """
    验证动态检测效果
    
    Args:
        env_path: Unity 环境路径
        num_episodes: 测试的 episode 数量
        steps_per_episode: 每个 episode 的步数
        visualize: 是否可视化
        verbose: 是否打印详细信息
    """
    
    # 环境配置
    env_cfg = EnvConfig(
        file_name=env_path,
        behavior_name="Navtest?team=0",
        no_graphics=not visualize,
        obs_size=187,
        lidar_dim=180,
    )
    
    env = UnityNavEnv(env_cfg)
    
    # 检测器
    detector = UniformMotionDetector(lidar_dim=180, history_len=10)
    
    # 可视化器
    visualizer = DetectionVisualizer(lidar_dim=180) if visualize else None
    
    # 统计
    stats = {
        'total_steps': 0,
        'detection_count': 0,
        'steps_with_detection': 0,
        'threat_levels': [],
        'velocities': [],
        'distances': [],
        'ttcs': [],
    }
    
    print("\n" + "="*60)
    print("Dynamic Detection Verification")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Visualize: {visualize}")
    print("="*60 + "\n")
    
    try:
        for ep in range(num_episodes):
            obs, info = env.reset()
            detector.reset()
            
            print(f"\n--- Episode {ep+1}/{num_episodes} ---")
            
            # 简单策略: 向目标方向移动，遇到障碍物转向
            action = np.array([0.5, 0.0], dtype=np.float32)
            
            for step in range(steps_per_episode):
                # 提取观测
                lidar = obs[:180]
                goal_dir_x = float(obs[180])
                goal_dir_z = float(obs[181])
                goal_dist = float(obs[182]) * 30.0
                goal_angle = float(obs[183]) * np.pi
                cur_v = float(obs[184])
                cur_w = float(obs[185])
                
                # 更新检测器
                detector.update(lidar, 0, 0, 0)
                
                # 检测动态物体
                features = detector.detect()
                detected_objects = detector.get_detected_objects()
                
                # 更新统计
                stats['total_steps'] += 1
                if detected_objects:
                    stats['steps_with_detection'] += 1
                    stats['detection_count'] += len(detected_objects)
                    
                    for obj in detected_objects:
                        stats['threat_levels'].append(obj['threat'])
                        stats['velocities'].append(np.sqrt(obj['radial_velocity']**2 + obj['tangential_velocity']**2))
                        stats['distances'].append(obj['distance'])
                        stats['ttcs'].append(obj['ttc'])
                
                # 打印信息
                if verbose and step % 20 == 0:
                    print(f"  Step {step}: Detected {len(detected_objects)} objects")
                    for i, obj in enumerate(detected_objects[:3]):  # 只显示前3个
                        print(f"    Object {i+1}: dist={obj['distance']:.2f}m, "
                              f"v_r={obj['radial_velocity']:.2f}, threat={obj['threat']:.2f}")
                
                # 可视化
                if visualize and step % 2 == 0:
                    visualizer.update(
                        lidar=lidar,
                        detected_objects=detected_objects,
                        robot_x=0, robot_z=0, robot_yaw=0,
                        goal_direction=(goal_dir_x, goal_dir_z)
                    )
                
                # 简单控制策略
                min_lidar = np.min(lidar) * 10.0
                if min_lidar < 1.0:
                    # 避障
                    front_lidar = lidar[60:120]
                    left_lidar = lidar[0:60]
                    right_lidar = lidar[120:180]
                    
                    if np.min(left_lidar) > np.min(right_lidar):
                        action = np.array([0.2, 0.8])  # 左转
                    else:
                        action = np.array([0.2, -0.8])  # 右转
                else:
                    # 朝目标前进
                    w = -np.sign(goal_angle) * min(abs(goal_angle), 1.0) * 0.5
                    v = 0.5 if abs(goal_angle) < 0.5 else 0.2
                    action = np.array([v, w])
                
                # 执行动作
                obs, reward, done, truncated, info = env.step(action)
                
                if done:
                    print(f"  Episode ended at step {step}: success={info.get('success', False)}, "
                          f"collision={info.get('collision', False)}")
                    break
            
            # Episode 结束后等待一下
            if visualize:
                plt.pause(0.5)
    
    finally:
        env.close()
    
    # 打印统计结果
    print("\n" + "="*60)
    print("Detection Statistics")
    print("="*60)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Steps with detection: {stats['steps_with_detection']} "
          f"({100*stats['steps_with_detection']/stats['total_steps']:.1f}%)")
    
    if stats['detection_count'] > 0:
        print(f"\nAverage detections per step (when detected): "
              f"{stats['detection_count']/stats['steps_with_detection']:.2f}")
        print(f"\nThreat level: mean={np.mean(stats['threat_levels']):.3f}, "
              f"max={np.max(stats['threat_levels']):.3f}")
        print(f"Velocity: mean={np.mean(stats['velocities']):.2f} m/s, "
              f"max={np.max(stats['velocities']):.2f} m/s")
        print(f"Distance: mean={np.mean(stats['distances']):.2f} m, "
              f"min={np.min(stats['distances']):.2f} m")
        print(f"TTC: mean={np.mean(stats['ttcs']):.2f} s, "
              f"min={np.min(stats['ttcs']):.2f} s")
        
        # 判断检测效果
        print("\n" + "-"*40)
        if stats['steps_with_detection'] / stats['total_steps'] > 0.3:
            print("✓ Detection is active - detecting objects frequently")
        else:
            print("⚠ Detection is sparse - may need parameter tuning")
        
        if np.mean(stats['velocities']) > 0.3:
            print("✓ Velocity estimation seems reasonable")
        else:
            print("⚠ Velocities are low - check if pedestrians are moving")
        
        if np.min(stats['distances']) < 3.0:
            print("✓ Detecting close objects - important for collision avoidance")
        else:
            print("⚠ No close detections - may miss nearby pedestrians")
    
    print("="*60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Verify dynamic object detection")
    parser.add_argument("--env", type=str, 
                       default=None,
                       help="Path to Unity environment (auto-detect if not specified)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to test")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # 自动检测环境路径
    env_path = args.env
    if env_path is None:
        import platform
        import os
        
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if platform.system() == "Linux":
            # Linux 环境
            linux_paths = [
                os.path.join(script_dir, "Corriidor_linux/Corridor_linux.x86_64"),
                "./Corriidor_linux/Corridor_linux.x86_64",
                "/home/dell/DRL_Navigation/Corriidor_linux/Corridor_linux.x86_64",
            ]
            for p in linux_paths:
                if os.path.exists(p):
                    env_path = p
                    break
        else:
            # Windows 环境
            win_paths = [
                r"D:\DRL_Navigation\Builds\Project_1.exe",
                os.path.join(script_dir, "Builds/Project_1.exe"),
            ]
            for p in win_paths:
                if os.path.exists(p):
                    env_path = p
                    break
        
        if env_path is None:
            print("Error: Could not find Unity environment. Please specify --env path")
            print(f"Searched paths for {platform.system()}:")
            if platform.system() == "Linux":
                for p in linux_paths:
                    print(f"  {p}")
            else:
                for p in win_paths:
                    print(f"  {p}")
            return
        
        print(f"Auto-detected environment: {env_path}")
    
    verify_detection(
        env_path=env_path,
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        visualize=not args.no_viz,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

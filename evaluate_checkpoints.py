"""
Checkpoint 筛选评估脚本

用于:
1. 批量评估 exp4 的 checkpoint，找出最佳模型
2. 生成详细的评估报告
3. 对比不同 checkpoint 的表现

使用方法:
python evaluate_checkpoints.py --exp exp4 --checkpoints 1100 1150 1200 1250 1300 --episodes 100
"""

import argparse
import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch

from unity_env import UnityNavEnv, EnvConfig


@dataclass
class EvalResult:
    checkpoint: str
    update: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    return_mean: float
    return_std: float
    length_mean: float
    length_std: float
    final_goal_dist_mean: float
    
    # 详细统计
    success_episodes: int
    collision_episodes: int
    timeout_episodes: int
    total_episodes: int


def load_model(checkpoint_path: str, device: torch.device):
    """加载模型 (兼容不同版本的 checkpoint)"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 检测模型类型
    model_type = ckpt.get("model_type", "unknown")
    ppo_cfg = ckpt.get("ppo_cfg", {})
    
    # 根据模型类型创建对应的模型
    if model_type == "cnn_gru_ppo_exp5":
        from train_ppo_exp5 import CNNGRUActorCriticWithAux, PPOConfig
        cfg = PPOConfig()
        model = CNNGRUActorCriticWithAux(
            lidar_dim=cfg.lidar_dim,
            low_dim=cfg.low_dim,
            action_dim=cfg.action_dim,
            use_aux_tasks=cfg.use_aux_tasks
        ).to(device)
    else:
        # 默认使用原有模型
        from train_ppo_cnn_gru_tensorboard import CNNGRUActorCritic, PPOConfig
        cfg = PPOConfig()
        model = CNNGRUActorCritic(
            lidar_dim=cfg.lidar_dim,
            low_dim=cfg.low_dim,
            action_dim=cfg.action_dim
        ).to(device)
    
    # 加载权重
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    
    return model, cfg, ckpt.get("update", 0)


def evaluate_checkpoint(
    env: UnityNavEnv,
    model,
    cfg,
    device: torch.device,
    num_episodes: int = 100,
    verbose: bool = True
) -> EvalResult:
    """评估单个 checkpoint"""
    
    # 根据配置导入正确的模块
    if hasattr(cfg, 'use_dynamic_features') and cfg.use_dynamic_features:
        from train_ppo_exp5 import (
            init_obs_history, init_seq_history, build_enhanced_obs,
            DynamicObstacleDetector
        )
        use_exp5 = True
    else:
        from train_ppo_cnn_gru_tensorboard import (
            init_obs_history, init_seq_history, build_enhanced_obs
        )
        use_exp5 = False
    
    returns = []
    lengths = []
    successes = []
    collisions = []
    timeouts = []
    final_goal_dists = []
    
    for ep in range(num_episodes):
        obs_np, info = env.reset()
        obs_hist = init_obs_history(obs_np, cfg.seq_len)
        
        if use_exp5:
            dynamic_detector = DynamicObstacleDetector(cfg.lidar_dim, history_len=5)
            enhanced_obs = build_enhanced_obs(obs_hist, cfg, dynamic_detector)
        else:
            enhanced_obs = build_enhanced_obs(obs_hist, cfg)
            dynamic_detector = None
        
        seq_hist = init_seq_history(enhanced_obs, cfg.seq_len)
        
        done = False
        ep_ret = 0.0
        ep_len = 0
        last_info = info
        
        while not done:
            seq_np = np.stack(seq_hist, axis=0).astype(np.float32)
            seq_tensor = torch.tensor(seq_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action_mean, _ = model.get_deterministic_action(seq_tensor)
                action_np = action_mean.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)
            
            obs_np, reward, done, truncated, info = env.step(action_np)
            ep_ret += reward
            ep_len += 1
            last_info = info
            
            if not done:
                obs_hist.append(obs_np.copy())
                if use_exp5 and dynamic_detector:
                    cur_v = float(obs_np[184])
                    cur_w = float(obs_np[185])
                    dynamic_detector.update(obs_np[:cfg.lidar_dim], cur_v, cur_w)
                    enhanced_obs = build_enhanced_obs(obs_hist, cfg, dynamic_detector)
                else:
                    enhanced_obs = build_enhanced_obs(obs_hist, cfg)
                seq_hist.append(enhanced_obs.copy())
        
        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(float(last_info.get("success", False)))
        collisions.append(float(last_info.get("collision", False)))
        timeouts.append(float(last_info.get("timeout", False)))
        final_goal_dists.append(float(last_info.get("goal_dist", np.nan)))
        
        if verbose and (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"success_rate={np.mean(successes):.3f}, "
                  f"collision_rate={np.mean(collisions):.3f}")
    
    return EvalResult(
        checkpoint="",  # 稍后填充
        update=0,       # 稍后填充
        success_rate=float(np.mean(successes)),
        collision_rate=float(np.mean(collisions)),
        timeout_rate=float(np.mean(timeouts)),
        return_mean=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        length_mean=float(np.mean(lengths)),
        length_std=float(np.std(lengths)),
        final_goal_dist_mean=float(np.nanmean(final_goal_dists)),
        success_episodes=int(np.sum(successes)),
        collision_episodes=int(np.sum(collisions)),
        timeout_episodes=int(np.sum(timeouts)),
        total_episodes=num_episodes
    )


def find_best_checkpoint(
    exp_name: str = "exp4",
    checkpoint_updates: Optional[List[int]] = None,
    num_episodes: int = 100,
    env_path: str = r"D:\DRL_Navigation\Builds\Project_1.exe",
    output_dir: str = "./eval_results"
):
    """
    找出最佳 checkpoint
    
    Args:
        exp_name: 实验名称 (exp1, exp2, exp3, exp4)
        checkpoint_updates: 要评估的 update 编号列表
        num_episodes: 每个 checkpoint 评估的 episode 数
        env_path: Unity 环境路径
        output_dir: 结果输出目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 默认评估 exp4 的关键 checkpoint
    if checkpoint_updates is None:
        checkpoint_dir = f"./checkpoints/cnn_gru_ppo_tb/{exp_name}"
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            checkpoint_updates = sorted([
                int(f.split("_")[-1].replace(".pt", ""))
                for f in files if f.endswith(".pt")
            ])
        else:
            print(f"Checkpoint directory not found: {checkpoint_dir}")
            return
    
    print(f"\n{'='*60}")
    print(f"Evaluating {len(checkpoint_updates)} checkpoints from {exp_name}")
    print(f"Updates to evaluate: {checkpoint_updates}")
    print(f"Episodes per checkpoint: {num_episodes}")
    print(f"{'='*60}\n")
    
    # 环境配置
    env_cfg = EnvConfig(
        file_name=env_path,
        behavior_name="Navtest?team=0",
        no_graphics=True,  # 无图形界面加速
        obs_size=187,
        lidar_dim=180,
        reach_goal_radius=0.5,
        max_steps=450,
    )
    
    env = UnityNavEnv(env_cfg)
    results: List[EvalResult] = []
    
    for update in checkpoint_updates:
        checkpoint_path = f"./checkpoints/cnn_gru_ppo_tb/{exp_name}/ppo_gru_update_{update:04d}.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n{'='*40}")
        print(f"Evaluating update {update}")
        print(f"{'='*40}")
        
        # 加载模型
        model, cfg, ckpt_update = load_model(checkpoint_path, device)
        
        # 评估
        start_time = time.time()
        result = evaluate_checkpoint(env, model, cfg, device, num_episodes)
        eval_time = time.time() - start_time
        
        result.checkpoint = checkpoint_path
        result.update = update
        
        results.append(result)
        
        print(f"\nResults for update {update}:")
        print(f"  Success Rate: {result.success_rate:.3f}")
        print(f"  Collision Rate: {result.collision_rate:.3f}")
        print(f"  Timeout Rate: {result.timeout_rate:.3f}")
        print(f"  Return: {result.return_mean:.3f} ± {result.return_std:.3f}")
        print(f"  Length: {result.length_mean:.1f} ± {result.length_std:.1f}")
        print(f"  Eval Time: {eval_time:.1f}s")
    
    env.close()
    
    # 分析结果
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # 按成功率排序
    results_sorted = sorted(results, key=lambda x: x.success_rate, reverse=True)
    
    print(f"\n{'Update':>8} {'Success':>10} {'Collision':>10} {'Timeout':>10} {'Return':>12} {'Length':>10}")
    print("-" * 60)
    for r in results_sorted:
        print(f"{r.update:>8} {r.success_rate:>10.3f} {r.collision_rate:>10.3f} "
              f"{r.timeout_rate:>10.3f} {r.return_mean:>10.3f}±{r.return_std:<4.2f} "
              f"{r.length_mean:>10.1f}")
    
    # 最佳 checkpoint
    best = results_sorted[0]
    print(f"\n{'='*60}")
    print(f"BEST CHECKPOINT: update {best.update}")
    print(f"{'='*60}")
    print(f"  Path: {best.checkpoint}")
    print(f"  Success Rate: {best.success_rate:.3f}")
    print(f"  Collision Rate: {best.collision_rate:.3f}")
    print(f"  Timeout Rate: {best.timeout_rate:.3f}")
    print(f"  Return: {best.return_mean:.3f} ± {best.return_std:.3f}")
    print(f"  Length: {best.length_mean:.1f} ± {best.length_std:.1f}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"eval_{exp_name}_{int(time.time())}.json")
    
    results_dict = {
        "exp_name": exp_name,
        "num_episodes": num_episodes,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_checkpoint": {
            "update": best.update,
            "path": best.checkpoint,
            "success_rate": best.success_rate,
        },
        "all_results": [asdict(r) for r in results_sorted]
    }
    
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return best


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints to find the best model")
    parser.add_argument("--exp", type=str, default="exp4", help="Experiment name (exp1, exp2, exp3, exp4)")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None,
                        help="Checkpoint updates to evaluate (e.g., 1100 1150 1200)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per checkpoint")
    parser.add_argument("--env", type=str, default=r"D:\DRL_Navigation\Builds\Project_1.exe",
                        help="Path to Unity environment")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory")
    
    args = parser.parse_args()
    
    find_best_checkpoint(
        exp_name=args.exp,
        checkpoint_updates=args.checkpoints,
        num_episodes=args.episodes,
        env_path=args.env,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

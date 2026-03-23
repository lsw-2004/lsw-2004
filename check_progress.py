#!/usr/bin/env python
"""
简单的训练进度检查器
解析 TensorBoard 日志或训练输出
"""
import os
import sys
import re
from collections import defaultdict

def parse_tb_logs(log_dir):
    """解析 TensorBoard 日志"""
    try:
        from tensorboard.backend.event_processing import event_file_loader
    except ImportError:
        return None
    
    # 找事件文件
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        return None
    
    latest = max([os.path.join(log_dir, f) for f in event_files], key=os.path.getmtime)
    
    data = defaultdict(list)
    for event in event_file_loader.EventFileLoader(latest).Load():
        if event.HasField('summary'):
            for v in event.summary.value:
                if v.HasField('simple_value'):
                    data[v.tag].append((event.step, v.simple_value))
    return data

def print_progress(data):
    """打印进度"""
    if not data:
        print("没有数据")
        return
    
    print("\n" + "="*60)
    print("训练进度摘要")
    print("="*60)
    
    # 成功率
    for tag in ['train_window/success_rate_50', 'eval/success_rate', 'train/episode_success']:
        if tag in data:
            vals = data[tag]
            if vals:
                latest = vals[-1][1]
                print(f"\n{tag}:")
                print(f"  最新值: {latest:.2%}")
                print(f"  数据点: {len(vals)}")
                
                # 趋势
                if len(vals) >= 10:
                    recent = sum(v for _, v in vals[-10:]) / 10
                    print(f"  最近10个平均: {recent:.2%}")
    
    # 碰撞率
    for tag in ['train_window/collision_rate_50', 'eval/collision_rate']:
        if tag in data:
            vals = data[tag]
            if vals:
                latest = vals[-1][1]
                print(f"\n{tag}:")
                print(f"  最新值: {latest:.2%}")
    
    # Entropy
    if 'train/entropy' in data:
        vals = data['train/entropy']
        if vals:
            print(f"\nEntropy: {vals[-1][1]:.4f}")
    
    print("\n" + "="*60)

def parse_log_text(text):
    """解析训练文本日志"""
    success_pattern = r'success=(\w+)'
    collision_pattern = r'collision=(\w+)'
    update_pattern = r'update=(\d+)'
    
    successes = []
    collisions = []
    
    for line in text.split('\n'):
        if '[train ep]' in line:
            s_match = re.search(success_pattern, line)
            c_match = re.search(collision_pattern, line)
            if s_match:
                successes.append(s_match.group(1) == 'True')
            if c_match:
                collisions.append(c_match.group(1) == 'True')
    
    if successes:
        total = len(successes)
        success_count = sum(successes)
        collision_count = sum(collisions)
        timeout_count = total - success_count - collision_count
        
        print("\n" + "="*60)
        print("从日志文本统计")
        print("="*60)
        print(f"总 episodes: {total}")
        print(f"成功: {success_count} ({success_count/total:.1%})")
        print(f"碰撞: {collision_count} ({collision_count/total:.1%})")
        print(f"超时: {timeout_count} ({timeout_count/total:.1%})")
        print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="./runs/cnn_gru_ppo_tb/exp5_v2")
    parser.add_argument("--logfile", help="训练日志文件路径")
    args = parser.parse_args()
    
    # 尝试读取 TensorBoard 日志
    if os.path.exists(args.logdir):
        data = parse_tb_logs(args.logdir)
        if data:
            print_progress(data)
    
    # 读取文本日志
    if args.logfile and os.path.exists(args.logfile):
        with open(args.logfile) as f:
            parse_log_text(f.read())

import gym
import os
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from pathlib import Path

# ================= 1. 配置路径 =================
# 根据你的截图，你的 exe 名字叫 Project_1.exe
# 请确保 "Builds" 文件夹和这个 train.py 在同一个目录下
# UNITY_BUILD_PATH = "./Builds/Project_1.exe" 
ROOT = Path(__file__).resolve().parent
UNITY_BUILD_PATH = ROOT / "Builds" / "Project_1.exe"
MODEL_SAVE_PATH = ROOT / "models"
LOG_DIR = ROOT / "logs"
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
assert UNITY_BUILD_PATH.exists(), f"找不到 Unity 可执行文件: {UNITY_BUILD_PATH}"




# 创建保存文件夹
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    print("正在连接 Unity 环境...")
    
    # 1. 启动 Unity 环境
    # worker_id=1 防止端口冲突
    # no_graphics=False 表示你会看到 Unity 窗口弹出来（训练时不要最小化它，可以被挡在后面）
    unity_env = UnityEnvironment(
        file_name=str(UNITY_BUILD_PATH),
        worker_id=1,
        base_port=5005,
        timeout_wait=300,
        no_graphics=False,
        additional_args=[
            "-screen-fullscreen", "0",
            "-screen-width", "1280",
            "-screen-height", "720",
            "-popupwindow",
            "-logFile", str(ROOT / "unity_player.log"),
        ],
    )

    print("Launching Unity...")
    unity_env.reset()
    print("Connected! behaviors =", list(unity_env.behavior_specs.keys()))

    # 2. 转换成 Gym 环境
    # allow_multiple_obs=False 会把我们在 C# 里写的 186 维数据作为一个向量传出来
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)

    print("环境连接成功！")
    print(f"观测空间维度: {env.observation_space.shape}") # 应该显示 (186,)
    print(f"动作空间维度: {env.action_space.shape}")      # 应该显示 (2,)

    # 3. 定义 PPO 模型
    # MlpPolicy: 适用于数值输入（雷达），不处理图像
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        batch_size=128,          # 稍微改大了点，训练更稳
        n_steps=2048,            # 每次更新收集的数据量
        gamma=0.99,              # 折扣因子
        tensorboard_log=LOG_DIR,  # 用于画图
        device="cuda"
    )

    # 4. 设置自动保存
    # 每 10000 步保存一次模型，防止电脑死机白跑
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=MODEL_SAVE_PATH, name_prefix="ppo_nav")
    
    # 5. 开始训练
    print("开始训练... (你可以使用 TensorBoard 查看进度)")
    # 建议先跑 20万步看看效果，毕设最终可能需要 100万步
    model.learn(total_timesteps=200000, callback=checkpoint_callback)

    # 6. 保存最终模型
    model.save(f"{MODEL_SAVE_PATH}/ppo_nav_final")
    print("训练完成！")

    # 关闭环境
    env.close()

if __name__ == '__main__':
    main()
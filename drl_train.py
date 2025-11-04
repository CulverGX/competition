# DRL_train.py
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from PyQt6.QtCore import QObject, pyqtSignal

from comsol_surrogate_train import SurrogateNN

# =========================================================
# 1️⃣ 加载 surrogate 模型
# =========================================================
checkpoint_path = "surrogate_model_optimized.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

input_dim = 13
output_dim = 6

model_surrogate = SurrogateNN(input_dim, output_dim)
model_surrogate.load_state_dict(checkpoint["model_state_dict"])
model_surrogate.eval()

x_mean = checkpoint["x_scaler_mean"]
x_scale = checkpoint["x_scaler_scale"]
y_mean = checkpoint["y_scalers_mean"]
y_scale = checkpoint["y_scalers_scale"]

# =========================================================
# 2️⃣ 定义奖励类和环境类
# =========================================================
class RewardCalculator:
    def __init__(self, w=None, smooth_alpha=0.9):
        self.smooth_alpha = smooth_alpha
        self.prev_reward = None
        self.w = w or {"w1": 1.0, "w2": 1.0, "w3": 1.0, "w4": 1.0, "w5": 1.0, "w6": 1.0}
        self.reward_max = sum(self.w.values())

    def compute(self, L_error, k_error, Ripple_ratio, Volume_ratio, Loss_ratio, Temp_ratio,
                L_target=1.0, k_target=1.0):
        L_err_norm = np.clip(L_error / (L_target + 1e-6), 0, 1)
        k_err_norm = np.clip(k_error / (k_target + 1e-6), 0, 1)
        Ripple_norm = np.clip(Ripple_ratio, 0, 1)
        Vol_norm = np.clip(Volume_ratio, 0, 1)
        Loss_norm = np.clip(Loss_ratio, 0, 1)
        Temp_norm = np.clip(Temp_ratio, 0, 1)

        reward = (
            self.w["w1"] * (1 - L_err_norm)
            + self.w["w2"] * (1 - k_err_norm)
            + self.w["w3"] * (1 - Ripple_norm)
            + self.w["w4"] * (1 - Vol_norm)
            + self.w["w5"] * (1 - Loss_norm)
            + self.w["w6"] * (1 - Temp_norm)
        )
        reward = np.clip(reward / self.reward_max, 0.0, 1.0)

        if self.prev_reward is None:
            smooth_reward = reward
        else:
            smooth_reward = self.smooth_alpha * self.prev_reward + (1 - self.smooth_alpha) * reward

        self.prev_reward = smooth_reward
        return smooth_reward


class SurrogateEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, mode="highfreq", target_L=10.0, target_k=0.95, custom_weights=None):
        super().__init__()
        self.input_dim = 13
        self.output_dim = 6

        # 空间定义
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.input_dim,), dtype=np.float32)

        self.state = np.zeros(self.input_dim, dtype=np.float32)
        self.target_L = target_L
        self.target_k = target_k
        self.mode = mode

        self.step_count = 0
        self.max_steps_per_episode = 50

        # 奖励函数权重
        default_weights = {
            "highfreq": dict(w1=1, w2=1, w3=0.2, w4=0.2, w5=0.25, w6=0.15),
            "highpower": dict(w1=1, w2=1, w3=0.2, w4=0.15, w5=0.2, w6=0.25)
        }
        self.w = custom_weights if custom_weights is not None else default_weights[self.mode]

        # ===== 创建 RewardCalculator =====
        self.reward_calc = RewardCalculator(w=self.w, smooth_alpha=0.9)

        # 参考值
        self.ref = dict(LCoil=2, Lmut=1.5, Ripple=3, Volume=10.19, Loss=18000, Temp=72)

        # ===== 自动加载数据计算上下限 =====
        input_cols = ["base_x", "base_z", "base_y", "g_1", "g_2", "g_3",
                      "thick_copper", "w_1", "w_2", "core_y", "r", "n", "I"]
        data = pd.read_csv("comsol_data.csv")[input_cols]

        self.input_bounds = {}
        for col in input_cols:
            mean, std = data[col].mean(), data[col].std()
            lower, upper = mean - 3 * std, mean + 3 * std
            # 保证非负
            lower = max(0.0, lower)
            self.input_bounds[col] = (float(lower), float(upper))

        self.lower_bounds = np.array([v[0] for v in self.input_bounds.values()], dtype=np.float32)
        self.upper_bounds = np.array([v[1] for v in self.input_bounds.values()], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 从实际物理范围随机初始化
        self.state = np.array([np.random.uniform(l, h) for l, h in zip(self.lower_bounds, self.upper_bounds)],
                              dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        # 限制动作范围在真实域内
        self.state = np.clip(self.state + action, self.lower_bounds, self.upper_bounds)
        self.state = np.maximum(self.state, 0.0)  # 非负约束

        # ===== 离散变量处理 =====
        # 假设 thick_copper 对应 state[1]，n 对应 state[2]
        thick_copper_idx = 6
        n_idx = 11

        allowed_thick = np.array([0.035, 0.07, 0.105])
        allowed_n = np.arange(1, 9)  # [1,2,3,4,5,6,7,8]

        # 将动作值映射到最近的离散值
        def discretize_value(raw_val, lower, upper, allowed_values):
            mapped_idx = int(np.clip(np.round((raw_val - lower) / (upper - lower) * (len(allowed_values) - 1)),
                                     0, len(allowed_values) - 1))
            return allowed_values[mapped_idx]

        self.state[thick_copper_idx] = discretize_value(
            self.state[thick_copper_idx],
            self.lower_bounds[thick_copper_idx],
            self.upper_bounds[thick_copper_idx],
            allowed_thick
        )

        self.state[n_idx] = discretize_value(
            self.state[n_idx],
            self.lower_bounds[n_idx],
            self.upper_bounds[n_idx],
            allowed_n
        )

        # ===== surrogate 预测 =====
        x_scaled = (self.state - x_mean) / x_scale
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_scaled = model_surrogate(x_tensor).cpu().numpy()[0]
        y_real = np.array([y_scaled[i] * y_scale[i] + y_mean[i] for i in range(self.output_dim)])

        LCoil, Lmut, Ripple, Volume, Loss, Temp = y_real
        k = Lmut / LCoil if LCoil != 0 else 0.0

        # ===== 误差项 =====
        L_error = abs(LCoil - self.target_L) / (abs(self.target_L) + 1e-12)
        k_error = abs(k - self.target_k) / (abs(self.target_k) + 1e-12)

        # ===== 比值 =====
        Ripple_ratio = float(Ripple / self.ref["Ripple"])
        Volume_ratio = float(Volume / self.ref["Volume"])
        Loss_ratio = float(Loss / self.ref["Loss"])
        Temp_ratio = float(Temp / self.ref["Temp"])

        # ===== 奖励函数 =====
        '''
        reward = (
            self.w["w1"] * np.exp(-L_error ** 2)
            + self.w["w2"] * np.exp(-k_error ** 2)
            - self.w["w3"] * np.tanh(Ripple_ratio)
            - self.w["w4"] * np.tanh(Volume_ratio)
            - self.w["w5"] * np.tanh(Loss_ratio)
            - self.w["w6"] * np.tanh(Temp_ratio)
        )
        '''


        reward = self.reward_calc.compute(
            L_error=L_error,
            k_error=k_error,
            Ripple_ratio=Ripple_ratio,
            Volume_ratio=Volume_ratio,
            Loss_ratio=Loss_ratio,
            Temp_ratio=Temp_ratio,
            L_target=self.target_L,
            k_target=self.target_k
        )


        # ===== 边界惩罚 =====
        boundary_penalty = np.mean(
            (self.state <= self.lower_bounds + 1e-6) | (self.state >= self.upper_bounds - 1e-6)
        )
        reward -= 0.5 * boundary_penalty

        reward = float(reward)
        self.step_count += 1
        terminated = bool(self.step_count >= self.max_steps_per_episode)
        truncated = False

        info = {
            "pred": y_real,
            "k": float(k),
            "reward": reward,
            "LCoil": float(LCoil),
            "Lmut": float(Lmut),
            "thick_copper": float(self.state[thick_copper_idx]),
            "n": int(self.state[n_idx])
        }

        return self.state, reward, terminated, truncated, info





# =========================================================
# 3️⃣ 封装为 PyQt 信号类
# =========================================================
class TrainingWorkerBackend(QObject):
    log_msg = pyqtSignal(str)
    epoch_result = pyqtSignal(int, float, float)  # epoch, loss, avg_reward
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal()
    top3_signal = pyqtSignal(list)  # 用于发出前三参数

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def run_training(self):
        # 配置部分
        mode_map = {"高频模式": "highfreq", "大功率模式": "highpower"}
        mode = mode_map.get(self.config.get("scene", "highfreq"), "highfreq")
        target_L = self.config.get("L_target", 2.0)
        target_k = self.config.get("M_target", 0.5)

        if mode == "highfreq":
            custom_weights = {"w1": 1, "w2": 1, "w3": 0.2, "w4": 0.2, "w5": 0.25, "w6": 0.15}
        elif mode == "highpower":
            custom_weights = {"w1": 1, "w2": 1, "w3": 0.2, "w4": 0.15, "w5": 0.2, "w6": 0.25}

        env = SurrogateEnv(mode=mode, target_L=target_L, target_k=target_k, custom_weights=custom_weights)

        env = DummyVecEnv([lambda: env])

        total_epochs = int(self.config.get("epochs", 50))
        ppo_model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=self.config.get("lr", 0.001),
            batch_size=self.config.get("batch_size", 64),
        )
        self.log_msg.emit("🚀 开始训练 PPO + Surrogate...")

        for epoch in range(1, total_epochs + 1):
            # 每一代训练
            ppo_model.learn(total_timesteps=1000)

            # 记录每一代指标
            L_errs, k_errs, Ripple_ratios, Volume_ratios, Loss_ratios, Temp_ratios = [], [], [], [], [], []
            LCoils, Lmuts = [], []

            obs = env.reset()
            step_rewards = []  # ✅ 存储每步奖励

            for step_i in range(10):
                action, _ = ppo_model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                step_rewards.append(float(reward))

                # ✅ 打印每步奖励到日志
                self.log_msg.emit(f"    Step {step_i + 1} reward = {float(reward):.6f}")

                # 提取 info 数据
                info_dict = info[0] if isinstance(info, list) else info
                y_pred = info_dict.get("pred", [0] * 6)
                k_val = info_dict.get("k", 0)

                # LCoil 和 Lmut
                LCoil = y_pred[0]
                Lmut = y_pred[1]

                # 计算各项误差与比值
                L_error = abs(LCoil - target_L) / (abs(target_L) + 1e-12)
                k_error = abs(k_val - target_k) / (abs(target_k) + 1e-12)
                Ripple_ratio = y_pred[2] / env.envs[0].ref["Ripple"]
                Volume_ratio = y_pred[3] / env.envs[0].ref["Volume"]
                Loss_ratio = y_pred[4] / env.envs[0].ref["Loss"]
                Temp_ratio = y_pred[5] / env.envs[0].ref["Temp"]

                # 记录指标
                L_errs.append(L_error)
                k_errs.append(k_error)
                Ripple_ratios.append(Ripple_ratio)
                Volume_ratios.append(Volume_ratio)
                Loss_ratios.append(Loss_ratio)
                Temp_ratios.append(Temp_ratio)
                LCoils.append(LCoil)
                Lmuts.append(Lmut)

            # ✅ 打印每轮奖励统计
            sum_reward = float(np.sum(step_rewards))
            avg_reward = float(np.mean(step_rewards))
            self.log_msg.emit(f"    Step rewards sum = {sum_reward:.6f}, avg = {avg_reward:.6f}")

            # 计算平均值
            avg_L = float(np.mean(L_errs))
            avg_k = float(np.mean(k_errs))
            avg_Ripple = float(np.mean(Ripple_ratios))
            avg_Volume = float(np.mean(Volume_ratios))
            avg_Loss = float(np.mean(Loss_ratios))
            avg_Temp = float(np.mean(Temp_ratios))
            avg_LCoil = float(np.mean(LCoils))
            avg_Lmut = float(np.mean(Lmuts))

            # 输出日志
            self.log_msg.emit(
                f"🔍 Epoch {epoch} avg metrics -> "
                f"L_err={avg_L:.3f}, k_err={avg_k:.3f}, "
                f"Ripple={avg_Ripple:.3f}, Vol={avg_Volume:.3f}, "
                f"Loss={avg_Loss:.3f}, Temp={avg_Temp:.3f}, "
                f"LCoil={avg_LCoil:.3f}, Lmut={avg_Lmut:.3f}"
            )

            self.epoch_result.emit(epoch, 0, avg_reward)
            self.progress.emit(int(epoch / total_epochs * 100))
            self.log_msg.emit(f"Epoch {epoch}/{total_epochs} - Avg Reward: {avg_reward:.4f}")

        # 保存模型
        ppo_model.save("ppo_surrogate_model")

        # 🌟 在训练结束后执行策略评估，传入训练时使用的 mode
        self.log_msg.emit("✅ PPO 模型已保存，开始策略评估")
        top3_results = self.evaluate_policy_top3(mode=mode)

        # 将前三组参数写入日志并发信号
        for i, (reward, params) in enumerate(top3_results, 1):
            # ✅ 格式化参数输出为三位小数
            formatted_params = "\n  ".join(
                [f"{k}: {v:.3f}" if isinstance(v, (float, int)) else f"{k}: {v}" for k, v in params.items()]
            )
            msg = f"🏆 Top-{i} | Reward={reward:.3f}\n  {formatted_params}"
            self.log_msg.emit(msg)

        # 发出信号给前端页面
        self.top3_results = top3_results
        self.top3_signal.emit(top3_results)

        self.finished_signal.emit()



    # =========================================================
    # 训练完成后的策略评估函数（取奖励最高的前三组参数）
    # =========================================================
    def evaluate_policy_top3(self, model_path="ppo_surrogate_model.zip", eval_episodes=100, mode="highfreq"):
        """
        评估 PPO 策略，返回奖励最高的前三组参数。
        mode: "highfreq" 或 "highpower"，保证权重与训练一致
        """
        # 根据模式选择权重
        if mode == "highfreq":
            custom_weights = {"w1": 1, "w2": 1, "w3": 0.2, "w4": 0.2, "w5": 0.25, "w6": 0.15}
        elif mode == "highpower":
            custom_weights = {"w1": 1, "w2": 1, "w3": 0.2, "w4": 0.15, "w5": 0.2, "w6": 0.25}
        else:
            raise ValueError(f"未知模式 {mode}，请选择 'highfreq' 或 'highpower'")

        # 创建环境
        env = SurrogateEnv(mode=mode, custom_weights=custom_weights)
        env = DummyVecEnv([lambda: env])

        # 加载 PPO 模型
        model = PPO.load(model_path)

        results = []  # 存储 (reward, params_dict)

        for ep in range(eval_episodes):
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result

            for _ in range(env.envs[0].max_steps_per_episode):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                if done:
                    break

            # 最终奖励
            final_reward = float(reward)

            # 从 info 提取离散后的 thick_copper 和 n
            info_dict = info[0] if isinstance(info, list) else info
            state = env.envs[0].state

            params = {
                "base_x": state[0],
                "base_z": state[1],
                "base_y": state[2],
                "g_1": state[3],
                "g_2": state[4],
                "g_3": state[5],
                "thick_copper": info_dict.get("thick_copper", float(state[6])),
                "w_1": state[7],
                "w_2": state[8],
                "core_y": state[9],
                "r": state[10],
                "n": info_dict.get("n", int(round(state[11]))),
                "extra": state[12],
            }

            results.append((final_reward, params))

        # 按奖励排序取前三
        top3 = sorted(results, key=lambda x: x[0], reverse=True)[:3]
        return top3

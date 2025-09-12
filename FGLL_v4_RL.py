# FG_vs_BP_CartPole_aligned.py
import os
import math
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import importlib

# =========================
# Gym 导入（带回退）
# =========================
GYM_MOD = None
for _name in ("gymnasium", "gym"):
    try:
        GYM_MOD = importlib.import_module(_name)
        break
    except Exception:
        continue
if GYM_MOD is None:
    raise ImportError("请先安装 gymnasium 或 gym：\n"
                      "pip install gymnasium[classic-control]\n"
                      "或\npip install gym[classic_control]")

# =========================
# 随机性与设备
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# =========================
# 全局对齐设置 & 超参数
# =========================
ENV_ID = "CartPole-v1"
GAMMA = 0.99
MAX_UPDATES = 60           # 参数更新的次数（两种方法相同）
UPD_EPISODES = 10          # ☆ 对齐：每次“更新”消耗的 episode 数
EVAL_EVERY = 2             # 每多少次更新打印一次日志（post-update 验证）

# —— Forward-Gradient (FG) —— #
# 2*ROLL_OUTS_FG (+ 可选1次新θ评估) ≈ UPD_EPISODES
ROLL_OUTS_FG = max(1, UPD_EPISODES // 2)   # 中心差分成对方向数
EPS_PERTURB = 0.02
LR_FG = 0.03

# —— REINFORCE (BP, 批量版) —— #
LR_BP = 3e-3
ENTROPY_COEF = 1e-3
GRAD_CLIP = 1.0
BATCH_EPISODES = UPD_EPISODES              # ☆ 对齐：每次更新的轨迹数

# 其他
HIDDEN_SIZES = (128, 128)
REWARD_CLIP = None
MOVING_AVG_W = 0.95

# 验证用的固定种子集（训练期间用于 post-update 验证 & 末尾最终评估）
VAL_SEEDS = list(range(90001, 90011))  # 10个验证种子


# =========================
# 版本无关的 Gym 帮助函数
# =========================
def try_seed_spaces(env, seed):
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
    except Exception:
        pass
    try:
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    except Exception:
        pass

def reset_env(env, seed=None):
    try:
        if seed is not None:
            out = env.reset(seed=seed)
        else:
            out = env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple):
        obs = out[0]
    else:
        obs = out
    return obs

def step_env(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, _ = out
        done = bool(terminated or truncated)
    else:
        obs, r, done, _ = out
        done = bool(done)
    return obs, r, done

def make_env(env_id=ENV_ID, seed=SEED):
    env = GYM_MOD.make(env_id)
    try_seed_spaces(env, seed)
    return env


# =========================
# 工具函数
# =========================
def set_global_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def discount_cumsum(rewards, gamma=GAMMA):
    ret = 0.0
    out = []
    for r in reversed(rewards):
        ret = r + gamma * ret
        out.append(ret)
    return list(reversed(out))

def moving_average(xs, w=MOVING_AVG_W):
    out = []
    ma = 0.0
    for i, x in enumerate(xs):
        ma = x if i == 0 else (w * ma + (1 - w) * x)
        out.append(ma)
    return out


# =========================
# 策略网络
# =========================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, act_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def act_sample(self, obs):
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    @torch.no_grad()
    def act_greedy(self, obs):
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())


# =========================
# 参数向量化（FG 用）
# =========================
def get_param_vector(model: nn.Module):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_param_vector(model: nn.Module, theta_vec: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(theta_vec[idx:idx+n].view_as(p))
        idx += n


# =========================
# 环境交互
# =========================
@torch.no_grad()
def rollout_return(env, policy: PolicyNet, greedy=False, reward_clip=REWARD_CLIP, seed=None):
    if seed is not None:
        set_global_seed(seed)
    obs = reset_env(env, seed=seed)
    done = False
    total_r = 0.0
    while not done:
        a = policy.act_greedy(obs) if greedy else policy.act_sample(obs)
        obs, r, done = step_env(env, a)
        if reward_clip is not None:
            r = max(min(r, reward_clip), -reward_clip)
        total_r += r
    return total_r


# =========================
# 统一评估（贪心 + 固定验证种子）
# =========================
@torch.no_grad()
def evaluate(env, policy, seeds=VAL_SEEDS):
    scores = []
    for sd in seeds:
        scores.append(rollout_return(env, policy, greedy=True, seed=sd))
    return float(np.mean(scores)), float(np.std(scores))


# =========================
# Forward-Gradient 训练（无 BP），对齐样本预算 + post-update 验证
# =========================
def train_forward_gradient(env_train, env_val, policy: PolicyNet,
                           updates=MAX_UPDATES, eps=EPS_PERTURB, lr=LR_FG,
                           rollouts=ROLL_OUTS_FG):
    theta = get_param_vector(policy).to(device)
    dim = theta.numel()
    train_returns = []   # 记录每次更新后，用新θ在训练环境上的一次采样回报（可选）
    val_means = []       # 记录每次更新后的贪心验证均值

    best_theta = theta.clone()
    best_val = -1e9

    for up in range(1, updates + 1):
        # ----- 梯度估计：中心差分的成对方向 -----
        g_est = torch.zeros_like(theta)
        for k in range(rollouts):
            v = torch.randn(dim, device=device)
            v = v / (v.norm() + 1e-8)

            set_param_vector(policy, theta + eps * v)
            J_plus = rollout_return(env_train, policy, greedy=False, seed=SEED + 20000 + up * 10 + k)

            set_param_vector(policy, theta - eps * v)
            J_minus = rollout_return(env_train, policy, greedy=False, seed=SEED + 30000 + up * 10 + k)

            g_est += ((J_plus - J_minus) / (2.0 * eps)) * v

        g_est /= rollouts

        # ----- 梯度上升 -----
        theta = theta + lr * g_est
        set_param_vector(policy, theta)

        # （可选）记录一次训练分布下的随机采样回报
        J_new = rollout_return(env_train, policy, greedy=False, seed=SEED + 40000 + up)
        train_returns.append(J_new)

        # ----- post-update 验证（贪心 + 固定验证种子）-----
        val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
        val_means.append(val_mean)

        if val_mean > best_val:
            best_val = val_mean
            best_theta = theta.clone()

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[FG] Update {up:3d} | train_return(sampled): {J_new:6.1f} | VAL(greedy): {val_mean:6.1f}")

    # 恢复最好参数
    set_param_vector(policy, best_theta)
    return train_returns, val_means, best_theta


# =========================
# 批量版 REINFORCE（带 baseline）+ 对齐样本预算 + post-update 验证
# =========================
def train_backprop_reinforce(env_train, env_val, policy: PolicyNet,
                             updates=MAX_UPDATES, batch_episodes=BATCH_EPISODES,
                             lr=LR_BP, gamma=GAMMA, entropy_coef=ENTROPY_COEF):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    train_returns = []  # 记录 batch 内各轨迹均值（用于观察）
    val_means = []      # 记录每次更新后的贪心验证均值

    best_state = copy.deepcopy(policy.state_dict())
    best_val = -1e9

    ep_counter = 0
    for up in range(1, updates + 1):
        batch_logps, batch_adv, batch_ent = [], [], []
        batch_returns = []

        # ===== 收集一个 batch（对齐样本预算）=====
        for i in range(batch_episodes):
            obs = reset_env(env_train, seed=SEED + 50000 + ep_counter)
            ep_counter += 1
            logps, entrs, rewards = [], [], []
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logps.append(dist.log_prob(a))
                entrs.append(dist.entropy())

                obs, r, done = step_env(env_train, a.item())
                if REWARD_CLIP is not None:
                    r = max(min(r, REWARD_CLIP), -REWARD_CLIP)
                rewards.append(r)

            Gs = discount_cumsum(rewards, gamma)
            Gs_t = torch.as_tensor(Gs, dtype=torch.float32, device=device)
            baseline = Gs_t.mean()
            adv = Gs_t - baseline

            batch_logps.append(torch.stack(logps))
            batch_ent.append(torch.stack(entrs))
            batch_adv.append(adv)
            batch_returns.append(float(sum(rewards)))

        # ===== 拼接 + 反传一步 =====
        logps_cat = torch.cat(batch_logps)
        adv_cat = torch.cat(batch_adv)
        ent_cat = torch.cat(batch_ent)

        if adv_cat.numel() > 1:
            adv_cat = (adv_cat - adv_cat.mean()) / (adv_cat.std() + 1e-8)

        loss_pg = -(logps_cat * adv_cat).mean()
        loss_ent = -entropy_coef * ent_cat.mean()
        loss = loss_pg + loss_ent

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()

        # batch 的训练均值（供参考）
        train_returns.append(float(np.mean(batch_returns)))

        # ===== post-update 验证（贪心 + 固定验证种子）=====
        with torch.no_grad():
            val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
        val_means.append(val_mean)

        if val_mean > best_val:
            best_val = val_mean
            best_state = copy.deepcopy(policy.state_dict())

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[BP] Update {up:3d} | batch_mean_return: {np.mean(batch_returns):6.1f} | VAL(greedy): {val_mean:6.1f}")

    policy.load_state_dict(best_state)
    return train_returns, val_means


# =========================
# 主流程
# =========================
def main():
    # 训练与验证环境（分离，避免状态干扰）
    env_fg_train = make_env(ENV_ID, seed=SEED)
    env_bp_train = make_env(ENV_ID, seed=SEED + 1)
    env_val = make_env(ENV_ID, seed=SEED + 2)  # 供两种方法共享的验证环境

    obs_dim = env_fg_train.observation_space.shape[0]
    act_dim = env_fg_train.action_space.n

    # 相同初始化的两份策略（公平对比）
    base_policy = PolicyNet(obs_dim, act_dim).to(device)
    fg_policy = copy.deepcopy(base_policy).to(device)
    bp_policy = copy.deepcopy(base_policy).to(device)

    print("\n=== 训练：Forward-Gradient（无 Backprop；样本预算对齐）===")
    fg_train_ret, fg_val_means, best_theta = train_forward_gradient(
        env_fg_train, env_val, fg_policy,
        updates=MAX_UPDATES, eps=EPS_PERTURB, lr=LR_FG, rollouts=ROLL_OUTS_FG
    )

    print("\n=== 训练：REINFORCE（批量；带 Backprop；样本预算对齐）===")
    bp_train_ret, bp_val_means = train_backprop_reinforce(
        env_bp_train, env_val, bp_policy,
        updates=MAX_UPDATES, batch_episodes=BATCH_EPISODES,
        lr=LR_BP, gamma=GAMMA, entropy_coef=ENTROPY_COEF
    )

    # 末尾最终评估（与训练期间相同的验证规则：贪心 + VAL_SEEDS）
    final_fg_mean, final_fg_std = evaluate(env_val, fg_policy, seeds=VAL_SEEDS)
    final_bp_mean, final_bp_std = evaluate(env_val, bp_policy, seeds=VAL_SEEDS)

    print("\n最终评估（贪心，固定验证种子集）:")
    print(f"  Forward-Gradient : {final_fg_mean:.2f} ± {final_fg_std:.2f}")
    print(f"  REINFORCE (BP)   : {final_bp_mean:.2f} ± {final_bp_std:.2f}")

    # 绘图（训练指标 & post-update 验证）
    plt.figure(figsize=(12, 5))

    # 左：训练指标
    plt.subplot(1, 2, 1)
    x_fg = np.arange(1, len(fg_train_ret) + 1)
    x_bp = np.arange(1, len(bp_train_ret) + 1)
    plt.plot(x_fg, fg_train_ret, alpha=0.3, label='FG train (sampled)')
    plt.plot(x_fg, moving_average(fg_train_ret), label='FG train EMA')
    plt.plot(x_bp, bp_train_ret, alpha=0.3, label='BP train (batch mean)')
    plt.plot(x_bp, moving_average(bp_train_ret), label='BP train EMA')
    plt.xlabel("Update #")
    plt.ylabel("Return")
    plt.title("Training metrics (not comparable across methods, only trend)")
    plt.legend()

    # 右：对齐的 post-update 验证（可横向比较）
    plt.subplot(1, 2, 2)
    x1 = np.arange(1, len(fg_val_means) + 1)
    x2 = np.arange(1, len(bp_val_means) + 1)
    plt.plot(x1, fg_val_means, label='FG VAL (greedy)')
    plt.plot(x2, bp_val_means, label='BP VAL (greedy)')
    plt.xlabel("Update #")
    plt.ylabel("Greedy Return (fixed seeds)")
    plt.title("Aligned post-update validation")
    plt.legend()

    plt.tight_layout()
    plt.savefig("rl_fg_vs_bp_aligned.png", dpi=150)
    plt.show()

    # 保存
    torch.save(best_theta.detach().cpu(), "fg_best_theta.pt")
    torch.save(fg_policy.state_dict(), "policy_fg.pt")
    torch.save(bp_policy.state_dict(), "policy_bp.pt")
    print("\n已保存：fg_best_theta.pt, policy_fg.pt, policy_bp.pt, rl_fg_vs_bp_aligned.png")


if __name__ == "__main__":
    main()
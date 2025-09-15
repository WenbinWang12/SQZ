# FG_vs_BP_CartPole_SPSA_aligned_epsgreedy_CI_adamparity_fixed_eval_batchmean.py
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
MAX_UPDATES = 200            # 参数更新的次数（两种方法相同）
UPD_EPISODES = 10           # ☆ 对齐：每次“更新”消耗的 episode 数（两侧一致且固定）
EVAL_EVERY = 2              # 每多少次更新打印一次日志（post-update 验证）
EPS_EVAL = 0.1             # ☆ 评估用 ε-greedy（两侧一致）

# —— 优化器公平性：两侧都用 Adam —— #
LR_ADAM = 3e-2            # ☆ 两侧相同学习率
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


# —— 前向学习（SPSA） —— #
assert UPD_EPISODES >= 2, "UPD_EPISODES 必须 >= 2"
if UPD_EPISODES % 2 != 0:
    print("[警告] 为严格对齐预算，UPD_EPISODES 应为偶数。将忽略最后 1 条轨迹。")
K_SPSA = UPD_EPISODES // 2  # 每次更新用 K 个 SPSA 方向（2K 条轨迹）
EPS_PERTURB = 0.02          # SPSA 扰动尺度

# —— REINFORCE（反向传播） —— #
GRAD_CLIP = 1.0
BATCH_EPISODES = UPD_EPISODES  # ☆ 对齐：每次更新的轨迹数
ENTROPY_COEF = 0.0             # ☆ 置 0，避免额外正则差异

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
# 参数向量化（SPSA 用）
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
# 环境交互（训练时）
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
        if REWARD_CLIP is not None:
            r = max(min(r, REWARD_CLIP), -REWARD_CLIP)
        total_r += r
    return total_r


# =========================
# 统一评估：ε-greedy（本地 RNG，可复现）+ 固定验证种子
# =========================
@torch.no_grad()
def evaluate_eps_greedy(env, policy, seeds=VAL_SEEDS, eps=EPS_EVAL):
    """
    评估策略：以概率 eps 采取随机动作（本地 rng 生成，按 seed 可复现），否则采取贪心 argmax。
    返回：均值、标准差
    """
    scores = []

    act_dim = getattr(env.action_space, "n", None)

    for sd in seeds:
        obs = reset_env(env, seed=sd)
        rng = np.random.default_rng(sd)  # 本地 rng，每个种子独立、可复现
        done = False
        total_r = 0.0

        while not done:
            explore = (rng.random() < eps)
            if explore:
                if act_dim is not None:
                    a = int(rng.integers(low=0, high=act_dim))
                else:
                    a = env.action_space.sample()
            else:
                logits = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                a = int(torch.argmax(logits, dim=-1).item())

            obs, r, done = step_env(env, a)
            if REWARD_CLIP is not None:
                r = max(min(r, REWARD_CLIP), -REWARD_CLIP)
            total_r += r

        scores.append(total_r)

    return float(np.mean(scores)), float(np.std(scores))


# =========================
# 前向学习（SPSA+Adam）训练：无 BP；样本预算对齐 + post-update 验证
# 左图训练指标：使用本次更新的 2K 条评估回报的“batch 均值”（不额外消耗样本预算）
# =========================
def train_spsa(env_train, env_val, policy: PolicyNet,
               updates=MAX_UPDATES, eps=EPS_PERTURB, lr=LR_ADAM, K=K_SPSA):
    theta = get_param_vector(policy).to(device)

    # ---- Adam 状态（与 BP 完全同参）----
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    b1, b2 = ADAM_BETAS
    eps_adam = ADAM_EPS
    t_step = 0

    train_returns = []   # 这里记录“本轮 2K 评估回报的均值”，用于左图
    val_means = []       # 每次更新后的 ε-greedy 验证均值（右图）
    val_stds  = []       # 每次更新后的 验证标准差（右图）

    best_theta = theta.clone()
    best_val = -1e9

    for up in range(1, updates + 1):
        # ----- SPSA 梯度估计 + 收集本轮的 2K 回报 -----
        g_est = torch.zeros_like(theta)
        batch_eval_returns = []  # 收集 J_plus/J_minus 以统计 batch 均值

        for k in range(K):
            delta = torch.randint_like(theta, low=0, high=2, device=device, dtype=torch.long)
            delta = delta.float().mul_(2.0).sub_(1.0)   # {0,1}->{-1,+1}

            set_param_vector(policy, theta + eps * delta)
            J_plus = rollout_return(env_train, policy, greedy=False,
                                    seed=SEED + 120000 + up * 1000 + k)
            batch_eval_returns.append(J_plus)

            set_param_vector(policy, theta - eps * delta)
            J_minus = rollout_return(env_train, policy, greedy=False,
                                     seed=SEED + 130000 + up * 1000 + k)
            batch_eval_returns.append(J_minus)

            g_est += ((J_plus - J_minus) / (2.0 * eps)) * delta

        g_est /= max(1, K)

        # ----- Adam 梯度上升（与 BP 同超参）-----
        t_step += 1
        m = b1 * m + (1.0 - b1) * g_est
        v = b2 * v + (1.0 - b2) * (g_est * g_est)
        m_hat = m / (1.0 - b1 ** t_step)
        v_hat = v / (1.0 - b2 ** t_step)
        theta = theta + lr * m_hat / (torch.sqrt(v_hat) + eps_adam)

        set_param_vector(policy, theta)

        # 左图训练指标：使用本轮 2K 回报的 batch 均值（不额外采样）
        train_returns.append(float(np.mean(batch_eval_returns)))

        # ----- post-update 验证（ε-greedy + 固定验证种子）-----
        val_mean, val_std = evaluate_eps_greedy(env_val, policy, seeds=VAL_SEEDS, eps=EPS_EVAL)
        val_means.append(val_mean)
        val_stds.append(val_std)

        if val_mean > best_val:
            best_val = val_mean
            best_theta = theta.clone()

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[SPSA+Adam] Update {up:3d} | batch_mean_return: {train_returns[-1]:6.1f} "
                  f"| VAL(eps={EPS_EVAL:.2f}): {val_mean:6.1f} ± {val_std:5.1f}")

    # 恢复最好参数
    set_param_vector(policy, best_theta)
    return train_returns, val_means, val_stds, best_theta


# =========================
# 批量版 REINFORCE（Adam；带 baseline；无熵正则）+ 对齐样本预算 + post-update 验证
# 左图训练指标：batch 内各轨迹回报的均值（与之前一致）
# =========================
def train_backprop_reinforce(env_train, env_val, policy: PolicyNet,
                             updates=MAX_UPDATES, batch_episodes=BATCH_EPISODES,
                             lr=LR_ADAM, gamma=GAMMA, entropy_coef=ENTROPY_COEF):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS)
    train_returns = []  # 记录 batch 内各轨迹均值（用于左图）
    val_means = []      # 每次更新后的 ε-greedy 验证均值（右图）
    val_stds  = []      # 每次更新后的 验证标准差（右图）

    best_state = copy.deepcopy(policy.state_dict())
    best_val = -1e9

    ep_counter = 0
    for up in range(1, updates + 1):
        batch_logps, batch_adv, batch_ent = [], [], []
        batch_returns = []

        # ===== 收集一个 batch（对齐样本预算）=====
        for i in range(batch_episodes):
            obs = reset_env(env_train, seed=SEED + 200000 + ep_counter)
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
        loss_ent = -entropy_coef * ent_cat.mean() if entropy_coef != 0.0 else 0.0
        loss = loss_pg + loss_ent

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()

        # 左图训练指标：batch 回报均值
        train_returns.append(float(np.mean(batch_returns)))

        # ===== post-update 验证（ε-greedy + 固定验证种子）=====
        with torch.no_grad():
            val_mean, val_std = evaluate_eps_greedy(env_val, policy, seeds=VAL_SEEDS, eps=EPS_EVAL)
        val_means.append(val_mean)
        val_stds.append(val_std)

        if val_mean > best_val:
            best_val = val_mean
            best_state = copy.deepcopy(policy.state_dict())

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[BP(Adam)]  Update {up:3d} | batch_mean_return: {train_returns[-1]:6.1f} "
                  f"| VAL(eps={EPS_EVAL:.2f}): {val_mean:6.1f} ± {val_std:5.1f}")

    policy.load_state_dict(best_state)
    return train_returns, val_means, val_stds


# =========================
# 主流程
# =========================
def main():
    # 训练与验证环境（分离，避免状态干扰）
    env_spsa_train = make_env(ENV_ID, seed=SEED)
    env_bp_train   = make_env(ENV_ID, seed=SEED + 1)
    env_val        = make_env(ENV_ID, seed=SEED + 2)  # 供两种方法共享的验证环境

    obs_dim = env_spsa_train.observation_space.shape[0]
    act_dim = env_spsa_train.action_space.n

    # 相同初始化的两份策略（公平对比）
    base_policy = PolicyNet(obs_dim, act_dim).to(device)
    spsa_policy = copy.deepcopy(base_policy).to(device)
    bp_policy   = copy.deepcopy(base_policy).to(device)

    print("\n=== 训练：SPSA + Adam（无 Backprop；样本预算对齐；左图=2K评估均值）===")
    spsa_train_ret, spsa_val_means, spsa_val_stds, best_theta = train_spsa(
        env_spsa_train, env_val, spsa_policy,
        updates=MAX_UPDATES, eps=EPS_PERTURB, lr=LR_ADAM, K=K_SPSA
    )

    print("\n=== 训练：REINFORCE + Adam（带 Backprop；样本预算对齐；左图=batch均值）===")
    bp_train_ret, bp_val_means, bp_val_stds = train_backprop_reinforce(
        env_bp_train, env_val, bp_policy,
        updates=MAX_UPDATES, batch_episodes=BATCH_EPISODES,
        lr=LR_ADAM, gamma=GAMMA, entropy_coef=ENTROPY_COEF
    )

    # 末尾最终评估（与训练期间相同的验证规则：ε-greedy + VAL_SEEDS）
    final_spsa_mean, final_spsa_std = evaluate_eps_greedy(env_val, spsa_policy, seeds=VAL_SEEDS, eps=EPS_EVAL)
    final_bp_mean,   final_bp_std   = evaluate_eps_greedy(env_val, bp_policy,   seeds=VAL_SEEDS, eps=EPS_EVAL)

    print("\n最终评估（ε-greedy，固定验证种子集）:")
    print(f"  SPSA+Adam (Forward) : {final_spsa_mean:.2f} ± {final_spsa_std:.2f}")
    print(f"  REINFORCE+Adam (BP) : {final_bp_mean:.2f} ± {final_bp_std:.2f}")

    # 绘图（训练指标 & post-update 验证 with CI）
    plt.figure(figsize=(12, 5))

    # 左：训练指标（两侧均为“当前更新所用样本的 batch 均值”）
    plt.subplot(1, 2, 1)
    x_fg = np.arange(1, len(spsa_train_ret) + 1)
    x_bp = np.arange(1, len(bp_train_ret) + 1)
    plt.plot(x_fg, spsa_train_ret, alpha=0.3, label='SPSA+Adam train (2K eval mean)')
    plt.plot(x_fg, moving_average(spsa_train_ret), label='SPSA+Adam train EMA')
    plt.plot(x_bp, bp_train_ret, alpha=0.3, label='BP(Adam) train (batch mean)')
    plt.plot(x_bp, moving_average(bp_train_ret), label='BP(Adam) train EMA')
    plt.xlabel("Update #")
    plt.ylabel("Return")
    plt.title("Training metrics (both are batch means)")
    plt.legend()

    # 右：对齐的 post-update 验证（均值 + 95% CI 阴影）
    plt.subplot(1, 2, 2)
    x1 = np.arange(1, len(spsa_val_means) + 1)
    x2 = np.arange(1, len(bp_val_means) + 1)

    n_val = max(1, len(VAL_SEEDS))
    spsa_ci = 1.96 * (np.array(spsa_val_stds) / math.sqrt(n_val))
    bp_ci   = 1.96 * (np.array(bp_val_stds)   / math.sqrt(n_val))

    plt.plot(x1, spsa_val_means, label=f'SPSA+Adam VAL (eps={EPS_EVAL:.2f})')
    plt.fill_between(x1,
                     np.array(spsa_val_means) - spsa_ci,
                     np.array(spsa_val_means) + spsa_ci,
                     alpha=0.2)

    plt.plot(x2, bp_val_means, label=f'BP(Adam) VAL (eps={EPS_EVAL:.2f})')
    plt.fill_between(x2,
                     np.array(bp_val_means) - bp_ci,
                     np.array(bp_val_means) + bp_ci,
                     alpha=0.2)

    plt.xlabel("Update #")
    plt.ylabel("Eps-greedy Return (fixed seeds)")
    plt.title("Aligned post-update validation (mean ± 95% CI)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("rl_spsa_vs_bp_aligned_epsgreedy_ci_adamparity_fixed_eval_batchmean.png", dpi=150)
    plt.show()

    # 保存
    torch.save(best_theta.detach().cpu(), "spsa_best_theta.pt")
    torch.save(spsa_policy.state_dict(), "policy_spsa.pt")
    torch.save(bp_policy.state_dict(), "policy_bp.pt")
    print("\n已保存：spsa_best_theta.pt, policy_spsa.pt, policy_bp.pt, "
          "rl_spsa_vs_bp_aligned_epsgreedy_ci_adamparity_fixed_eval_batchmean.png")


if __name__ == "__main__":
    main()

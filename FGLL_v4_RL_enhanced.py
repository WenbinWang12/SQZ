# FG_Enhanced_vs_BP_CartPole.py
import os
import math
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    raise ImportError("请先安装 gymnasium 或 gym：pip install gymnasium[classic-control] 或 pip install gym[classic_control]")

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
# 超参数（可按需调整）
# =========================
ENV_ID = "CartPole-v1"
GAMMA = 0.99
MAX_EPISODES = 800            # 训练回合数（适当增大，配合更稳的 FG）
EVAL_EVERY = 20

# —— FG（Forward-Gradient）增强选项 —— #
ROLL_OUTS_FG = 16             # 方向对数（一次方向包含 +v/-v），建议 8~32
EPISODES_PER_DIRECTION = 4    # 每个方向评估的回合数（均值），建议 2~8
EPS_PERTURB = 0.03            # 扰动幅度 ε（配合 CRN 可用 0.02~0.05）
LR_FG = 0.04                  # Adam 初始学习率（用于参数向量）
FG_USE_ORTHO_DIRECTIONS = True # 使用 QR 正交化方向集
FG_DIRECTION_TYPE = "gaussian" # "gaussian" 或 "rademacher"
FITNESS_SHAPING = "rank"      # "rank"（推荐）或 "zscore" 或 "none"
GRAD_NORM_CLIP = 5.0          # 对 g_est 做范数裁剪与归一化（更新前）
DETERMINISTIC_ACTION_DURING_FG = True  # FG 评估使用 argmax 动作
UNFREEZE_SCHEDULE = [         # 逐层解冻策略（按 episode）
    ("last", 0),              # 从 ep >= 0 只扰动最后一层
    ("last2", 200),           # 从 ep >= 200 扰动最后两层
    ("all", 500),             # 从 ep >= 500 扰动全部层
]

# —— REINFORCE（BP） —— #
LR_BP = 1e-2                  # 学习率
ENTROPY_COEF = 0.01           # 熵正则权重
BP_GRAD_CLIP = 1.0            # 梯度裁剪

# 其他
HIDDEN_SIZES = (128, 128)
REWARD_CLIP = None            # 也可设为 1.0 做回报裁剪
MOVING_AVG_W = 0.95           # 绘图 EMA 系数


# =========================
# 版本无关的 Gym 帮助函数
# =========================
def try_seed_spaces(env, seed):
    """给 action/observation space 设种子（如果支持）。"""
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
    """
    统一重置接口：
    - 新 API: obs, info = env.reset(seed=seed)
    - 旧 API: obs = env.reset()
    返回 obs
    """
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
    """
    统一 step 接口：
    - 新 API: obs, r, terminated, truncated, info
    - 旧 API: obs, r, done, info
    返回 obs, r, done
    """
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

def softmax_logits_to_dist(logits):
    return torch.distributions.Categorical(logits=logits)

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


# =========================
# 参数向量化与切片映射
# =========================
def build_param_slices(model: nn.Module):
    """
    返回：
    - flat 参数向量
    - 切片列表: [(name, shape, start, end, param_ref), ...]
    - 线性层参数名集合（按顺序）
    """
    vecs = []
    slices = []
    names_linear = []  # 记录线性层（按前向顺序）
    start = 0
    linear_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear_layers.append(m)
    # 构建 name->module 的映射用于识别最后/倒数第二线性层
    last_linear = linear_layers[-1] if len(linear_layers) >= 1 else None
    last2_linear = linear_layers[-2] if len(linear_layers) >= 2 else None

    for name, p in model.named_parameters():
        n = p.numel()
        vecs.append(p.data.view(-1))
        end = start + n
        slices.append((name, p.shape, start, end, p))
        start = end
        # 记录属于线性层的参数名（便于子空间掩码）
        # 通过参数对象的父模块判断
        parent_module_name = name.rsplit('.', 1)[0] if '.' in name else ''
        # 直接用对象匹配更靠谱：逐层查找其所属 Linear
        owner = None
        for mod in linear_layers:
            for pname, pp in mod.named_parameters(recurse=False):
                if pp is p:
                    owner = mod
                    break
            if owner is not None:
                break
        if owner is not None:
            names_linear.append((name, owner))
    flat = torch.cat(vecs)
    return flat, slices, names_linear, last_linear, last2_linear

def get_param_vector(model: nn.Module):
    flat, *_ = build_param_slices(model)
    return flat

def set_param_vector(model: nn.Module, theta_vec: torch.Tensor):
    _, slices, *_ = build_param_slices(model)
    with torch.no_grad():
        for name, shape, s, e, pref in slices:
            pref.data.copy_(theta_vec[s:e].view(shape))

def get_param_dim(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def make_mask(model: nn.Module, scope: str = "last"):
    """
    scope: "last" | "last2" | "all"
    返回与参数向量等长的 0/1 掩码（float）
    """
    theta, slices, names_linear, last_linear, last2_linear = build_param_slices(model)
    mask = torch.zeros_like(theta)

    if scope == "all" or last_linear is None:
        mask[:] = 1.0
        return mask

    def mark_layer(mod):
        for name, shape, s, e, pref in slices:
            # 找到属于该层的 param
            owner = None
            for pname, pp in mod.named_parameters(recurse=False):
                if pp is pref:
                    owner = mod
                    break
            if owner is not None:
                mask[s:e] = 1.0

    if scope == "last":
        mark_layer(last_linear)
    elif scope == "last2":
        mark_layer(last_linear)
        if last2_linear is not None:
            mark_layer(last2_linear)
    else:
        mask[:] = 1.0
    return mask


# =========================
# 环境交互
# =========================
@torch.no_grad()
def rollout_return(env, policy: PolicyNet, seed=None, reward_clip=REWARD_CLIP, deterministic=False):
    if seed is not None:
        set_global_seed(seed)
    obs = reset_env(env, seed=seed)
    done = False
    total_r = 0.0
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(obs_t)
        if deterministic:
            a = torch.argmax(logits, dim=-1).item()
        else:
            dist = softmax_logits_to_dist(logits)
            a = dist.sample().item()
        obs, r, done = step_env(env, a)
        if reward_clip is not None:
            r = max(min(r, reward_clip), -reward_clip)
        total_r += r
    return total_r


# =========================
# 方向生成（正交化 / Rademacher）
# =========================
def sample_directions(dim, k, mask=None, ortho=True, kind="gaussian"):
    """
    返回 (k, dim) 的单位向量集合（在掩码子空间内近似正交）
    """
    if mask is None:
        mask = torch.ones(dim, device=device)
    else:
        mask = mask.to(device)

    # 先在掩码子空间内采样
    if kind == "rademacher":
        V = torch.empty(k, dim, device=device).bernoulli_(0.5).mul_(2).sub_(1)  # ±1
    else:
        V = torch.randn(k, dim, device=device)

    V *= mask  # 限制在子空间
    # 若某些向量全零（掩码太小），补上掩码内的小随机
    for i in range(k):
        if torch.allclose(V[i].abs().sum(), torch.tensor(0.0, device=device)):
            V[i] = torch.randn(dim, device=device) * mask

    # 正交化（对掩码子空间的投影进行 QR）
    if ortho:
        # 做 QR 需要矩阵 (dim x k)
        M = V.T  # (dim, k)
        # 若掩码子空间维度 < k，QR 也能返回近似正交的列子空间
        Q, _ = torch.linalg.qr(M, mode='reduced')  # (dim, r)
        V = Q.T  # (r, dim)
        # 如果 r < k，补齐
        if V.shape[0] < k:
            extra = torch.randn(k - V.shape[0], dim, device=device) * mask
            V = torch.cat([V, extra], dim=0)

    # 归一化
    V = V / (V.norm(dim=1, keepdim=True) + 1e-8)
    return V[:k]  # (k, dim)


# =========================
# FG 训练（增强版，无 BP）
# =========================
def train_forward_gradient_enhanced(env, model_template: PolicyNet, episodes=MAX_EPISODES):
    """
    增强点：
    - CRN（+/- 共用随机数流）
    - 每方向多回合均值
    - 正交/拉德马赫方向
    - rank-based / zscore fitness shaping
    - 子空间扰动（last/last2/all）+ 逐层解冻
    - 梯度归一化 + Adam 自适应步长
    - FG 评估使用确定性动作（可选）
    """
    # 初始参数向量
    theta_vec = get_param_vector(model_template).to(device).detach().clone()
    dim = theta_vec.numel()

    # 一个“影子模型”用于把向量写回评估
    shadow = copy.deepcopy(model_template).to(device)
    set_param_vector(shadow, theta_vec)

    # Adam 优化器（直接对参数向量做优化）
    theta_param = nn.Parameter(theta_vec.clone().detach())
    optimizer = torch.optim.Adam([theta_param], lr=LR_FG)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(episodes, 1))

    returns = []
    best_theta = theta_param.data.clone()
    best_score = -1e9

    # 当前掩码（逐层解冻）
    def current_scope(ep):
        sc = UNFREEZE_SCHEDULE[0][0]
        for scope_name, start_ep in UNFREEZE_SCHEDULE:
            if ep >= start_ep:
                sc = scope_name
        return sc

    for ep in range(1, episodes + 1):
        # 按进度选择子空间
        scope = current_scope(ep)
        mask = make_mask(shadow, scope=scope).to(device)

        # 采样一批方向（在子空间内近似正交）
        K = ROLL_OUTS_FG
        V = sample_directions(dim, K, mask=mask, ortho=FG_USE_ORTHO_DIRECTIONS, kind=FG_DIRECTION_TYPE)  # (K, dim)

        deltas = []
        # 评估每个方向的中心差分（CRN + 多回合均值）
        with torch.no_grad():
            for k in range(K):
                v = V[k]
                # 同方向内共用同一套随机数流（EPISODES_PER_DIRECTION 个 seed）
                seed_base = SEED + 100000 + ep * 1000 + k * 50

                # theta + eps v
                set_param_vector(shadow, theta_param.data + EPS_PERTURB * v)
                plus_scores = []
                for i in range(EPISODES_PER_DIRECTION):
                    plus_scores.append(
                        rollout_return(env, shadow,
                                       seed=seed_base + i,
                                       reward_clip=REWARD_CLIP,
                                       deterministic=DETERMINISTIC_ACTION_DURING_FG)
                    )
                J_plus = float(np.mean(plus_scores))

                # theta - eps v
                set_param_vector(shadow, theta_param.data - EPS_PERTURB * v)
                minus_scores = []
                for i in range(EPISODES_PER_DIRECTION):
                    # 同样的 seed 流（CRN）
                    minus_scores.append(
                        rollout_return(env, shadow,
                                       seed=seed_base + i,
                                       reward_clip=REWARD_CLIP,
                                       deterministic=DETERMINISTIC_ACTION_DURING_FG)
                    )
                J_minus = float(np.mean(minus_scores))

                delta = (J_plus - J_minus) / (2.0 * EPS_PERTURB)  # 方向导数估计（标量）
                deltas.append(delta)

        deltas_t = torch.tensor(deltas, dtype=torch.float32, device=device)  # (K,)

        # fitness shaping（把方向导数转成稳定的权重）
        if FITNESS_SHAPING == "rank":
            ranks = torch.argsort(torch.argsort(deltas_t))
            w = ranks.float() / max(K - 1, 1) - 0.5  # 居中到 [-0.5, 0.5]
        elif FITNESS_SHAPING == "zscore":
            w = (deltas_t - deltas_t.mean()) / (deltas_t.std() + 1e-8)
        else:
            w = deltas_t

        # 聚合估计梯度（方向向量的加权和）
        g_est = torch.sum(w.view(-1, 1) * V, dim=0)  # (dim,)
        # 仅在掩码子空间更新
        g_est = g_est * mask

        # 范数裁剪 + 归一化
        g_norm = g_est.norm()
        if GRAD_NORM_CLIP is not None and g_norm > GRAD_NORM_CLIP:
            g_est = g_est * (GRAD_NORM_CLIP / (g_norm + 1e-8))
        # 也可以按单位向量更新（注释掉以仅使用上面的裁剪）
        g_est = g_est / (g_est.norm() + 1e-8)

        # Adam 进行“梯度上升”：手动写入 grad = -g_est，然后 step（因为 PyTorch 的优化器默认做梯度下降）
        optimizer.zero_grad(set_to_none=True)
        theta_param.grad = -g_est  # 上升
        optimizer.step()
        scheduler.step()

        # 将更新后的参数写回 shadow 模型并评估一次（记录回报曲线）
        set_param_vector(shadow, theta_param.data)
        J_eval = rollout_return(env, shadow,
                                seed=SEED + 200000 + ep,
                                reward_clip=REWARD_CLIP,
                                deterministic=False)  # 评估时可用随机策略
        returns.append(J_eval)
        if J_eval > best_score:
            best_score = J_eval
            best_theta = theta_param.data.clone()

        if ep % EVAL_EVERY == 0 or ep == 1:
            avg20 = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
            print(f"[FG+] Ep {ep:4d} | scope={scope:>5s} | Return: {J_eval:.1f} | avg_last_20={avg20:.1f}")

    # 恢复最好参数
    set_param_vector(shadow, best_theta)
    return returns, best_theta, shadow


# =========================
# REINFORCE（BP）
# =========================
def train_backprop_reinforce(env, policy: PolicyNet, episodes=MAX_EPISODES,
                             lr=LR_BP, gamma=GAMMA, entropy_coef=ENTROPY_COEF):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    returns = []
    best_state = copy.deepcopy(policy.state_dict())
    best_score = -1e9

    for ep in range(1, episodes + 1):
        obs = reset_env(env, seed=SEED + 300000 + ep)
        logps = []
        entropies = []
        rewards = []
        done = False

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(obs_t)
            dist = softmax_logits_to_dist(logits)
            a = dist.sample()
            logp = dist.log_prob(a)
            entropy = dist.entropy().mean()

            obs, r, done = step_env(env, a.item())
            if REWARD_CLIP is not None:
                r = max(min(r, REWARD_CLIP), -REWARD_CLIP)

            logps.append(logp)
            entropies.append(entropy)
            rewards.append(r)

        # 回报（折扣）并标准化
        Gs = discount_cumsum(rewards, gamma)
        Gs_t = torch.as_tensor(Gs, dtype=torch.float32, device=device)
        if len(Gs_t) > 1:
            Gs_t = (Gs_t - Gs_t.mean()) / (Gs_t.std() + 1e-8)

        logps_t = torch.stack(logps)
        entropies_t = torch.stack(entropies) if len(entropies) > 0 else torch.tensor(0.0, device=device)

        loss_pg = -(logps_t * Gs_t).sum()
        loss_ent = -entropy_coef * entropies_t.mean()
        loss = loss_pg + loss_ent

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), BP_GRAD_CLIP)
        optimizer.step()

        ret = float(sum(rewards))
        returns.append(ret)
        if ret > best_score:
            best_score = ret
            best_state = copy.deepcopy(policy.state_dict())

        if ep % EVAL_EVERY == 0 or ep == 1:
            avg20 = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
            print(f"[BP ] Ep {ep:4d} | Return: {ret:.1f} | avg_last_20={avg20:.1f}")

    policy.load_state_dict(best_state)
    return returns


# =========================
# 评估
# =========================
@torch.no_grad()
def evaluate(env, policy, n_episodes=10):
    scores = []
    for i in range(n_episodes):
        scores.append(rollout_return(env, policy, seed=SEED + 400000 + i, deterministic=False))
    return float(np.mean(scores)), float(np.std(scores))


# =========================
# 主流程
# =========================
def main():
    # 两个独立环境用于两条训练线
    env_fg = make_env(ENV_ID, seed=SEED)
    env_bp = make_env(ENV_ID, seed=SEED + 1)

    obs_dim = env_fg.observation_space.shape[0]
    act_dim = env_fg.action_space.n

    # 相同初始化的两份策略（公平对比）
    base_policy = PolicyNet(obs_dim, act_dim).to(device)
    fg_init = copy.deepcopy(base_policy).to(device)
    bp_policy = copy.deepcopy(base_policy).to(device)

    print("\n=== 训练：Forward-Gradient（增强版，无 Backprop）===")
    fg_returns, best_theta, fg_policy = train_forward_gradient_enhanced(env_fg, fg_init, episodes=MAX_EPISODES)

    print("\n=== 训练：REINFORCE（带 Backprop）===")
    bp_returns = train_backprop_reinforce(env_bp, bp_policy, episodes=MAX_EPISODES)

    # 评估
    eval_env = make_env(ENV_ID, seed=SEED + 2)
    fg_mean, fg_std = evaluate(eval_env, fg_policy)
    bp_mean, bp_std = evaluate(eval_env, bp_policy)

    print(f"\n评估（10回合平均±标准差）:")
    print(f"  Forward-Gradient+ : {fg_mean:.2f} ± {fg_std:.2f}")
    print(f"  REINFORCE (BP)     : {bp_mean:.2f} ± {bp_std:.2f}")

    # 绘图
    plt.figure(figsize=(10, 5))
    x1 = np.arange(1, len(fg_returns) + 1)
    x2 = np.arange(1, len(bp_returns) + 1)
    plt.plot(x1, fg_returns, alpha=0.3, label='FG+ raw')
    plt.plot(x1, moving_average(fg_returns), label='FG+ EMA')
    plt.plot(x2, bp_returns, alpha=0.3, label='BP raw')
    plt.plot(x2, moving_average(bp_returns), label='BP EMA')
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("CartPole-v1: Forward-Gradient (Enhanced) vs. REINFORCE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rl_fg_enhanced_vs_bp.png", dpi=150)
    plt.show()

    # 保存
    torch.save(best_theta.detach().cpu(), "fg_plus_best_theta.pt")
    torch.save(fg_policy.state_dict(), "policy_fg_plus.pt")
    torch.save(bp_policy.state_dict(), "policy_bp.pt")
    print("\n已保存：fg_plus_best_theta.pt, policy_fg_plus.pt, policy_bp.pt, rl_fg_enhanced_vs_bp.png")


if __name__ == "__main__":
    main()
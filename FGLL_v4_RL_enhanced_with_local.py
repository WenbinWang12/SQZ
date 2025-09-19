# fg_bp_cartpole_gridsearch_local_forward.py
import os
import math
import csv
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import importlib
from datetime import datetime

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
    raise ImportError(
        "请先安装 gymnasium 或 gym：\n"
        "pip install gymnasium[classic-control]\n或\npip install gym[classic_control]"
    )

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
print(f"[Info] 使用设备: {device}")

# =========================
# 全局对齐设置 & 超参数（硬对齐）
# =========================
ENV_ID = "CartPole-v1"
GAMMA = 0.99
MAX_UPDATES = 60             # 两侧一致
UPD_EPISODES = 10            # 两侧一致：每次更新消耗的轨迹数
EVAL_EVERY = 2               # 打印频率
REWARD_CLIP = None           # 如需更稳，可设为 5
HIDDEN_SIZES = (128, 128)
MOVING_AVG_W = 0.95
VAL_SEEDS = list(range(90001, 90011))  # 固定验证种子
STEADY_K = 5                 # 评分时取尾部 K 次验证均值

# =========================
# “软对齐”搜索空间（可按需扩/缩）
# =========================
# —— Forward Learning（原 SPSA 的 eps / lr；我们仍然用黑盒扰动估计，但目标已包含 local loss）
FL_LR_GRID = [0.00018, 0.00019, 0.000195, 0.0002]
FL_EPS_GRID = [0.041, 0.042, 0.043, 0.044]
assert UPD_EPISODES >= 2, "UPD_EPISODES 必须 >= 2"
if UPD_EPISODES % 2 != 0:
    print("[警告] 为严格对齐预算，UPD_EPISODES 应为偶数。将忽略最后 1 条轨迹。")
K_FL = UPD_EPISODES // 2

# —— REINFORCE（backprop）
BP_LR_GRID = [0.009, 0.0095, 0.01, 0.011, 0.012]
BP_ENTROPY_GRID = [0.0006, 0.00065, 0.0007, 0.00075, 0.0008]

# =========================
# Local Loss 配置（可自由开关/加权）
# =========================
# 每个条目若 weight=0.0 则等价于关闭
# ent_target 需要 target（n_actions 的最大熵为 log(n_actions)）
LOCAL_LOSS_CFG = {
    "act_l2":     {"weight": 0.05},                # 激活 L2 正则（越小越好）
    "decor":      {"weight": 0.02},                # 同层去相关（越小越好）
    "slow":       {"weight": 0.02},                # 激活慢变（越小越好）
    "ent_target": {"weight": 0.02, "target": 0.65} # 策略熵目标（越接近越好）
}
GLOBAL_REWARD_WEIGHT = 1.0  # w_R

def is_cfg_enabled(cfg: dict):
    if cfg is None:
        return False
    for k, v in cfg.items():
        if isinstance(v, dict) and v.get("weight", 0.0) != 0.0:
            return True
    return False

# =========================
# 工具：种子/环境/绘图
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
    out, ma = [], 0.0
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
        layers, last = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, act_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()
        self.act_dim = act_dim  # 供熵上限等计算使用

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
    def _forward_with_acts(self, x):
        """前向并记录每个隐藏层激活（ReLU 之后）"""
        acts = []
        out = x
        i = 0
        while i < len(self.net):
            layer = self.net[i]
            if isinstance(layer, nn.Linear):
                out = layer(out)
                if i + 1 < len(self.net) and isinstance(self.net[i+1], nn.ReLU):
                    relu = self.net[i+1]
                    out_relu = relu(out)
                    acts.append(out_relu)  # 保存该层激活
                    out = out_relu
                    i += 2
                    continue
            else:
                out = layer(out)
            i += 1
        logits = out
        return logits, acts

    @torch.no_grad()
    def act_sample(self, obs):
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        # 相对平移不改变 softmax，增稳；再做温和兜底
        logits = logits - torch.amax(logits, dim=-1, keepdim=True)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())

    @torch.no_grad()
    def act_greedy(self, obs):
        logits = self.forward(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        logits = logits - torch.amax(logits, dim=-1, keepdim=True)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return int(torch.argmax(logits, dim=-1).item())

# =========================
# 参数向量化（Forward Learning）
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
# 交互与评估 + 局部损失计算
# =========================
@torch.no_grad()
def rollout_metrics(env, policy: PolicyNet, greedy=False, reward_clip=REWARD_CLIP, seed=None,
                    need_local=True):
    """
    同一条轨迹同时收集：
      - total return
      - 每步 logits 的熵（通过 logits_arr）
      - 每层激活的时间序列（acts_traces）
    """
    if seed is not None:
        set_global_seed(seed)
    obs = reset_env(env, seed=seed)
    done, total_r = False, 0.0
    logits_list = []
    layer_act_traces = []  # list per layer -> list of (H,)
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if need_local:
            logits, acts = policy._forward_with_acts(obs_t)
            if not layer_act_traces:
                layer_act_traces = [[] for _ in acts]
            for li, a in enumerate(acts):
                layer_act_traces[li].append(a.squeeze(0).detach().cpu().numpy())
        else:
            logits = policy(obs_t)

        # 相对平移与兜底，避免极端数值
        logits = logits - torch.amax(logits, dim=-1, keepdim=True)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        logits_list.append(logits.detach().cpu().numpy())
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample() if not greedy else torch.argmax(logits, dim=-1)
        obs, r, done = step_env(env, int(a.item()))
        if reward_clip is not None:
            r = max(min(r, reward_clip), -reward_clip)
        total_r += r

    if not need_local:
        return total_r, None, None
    logits_arr = np.concatenate(logits_list, axis=0)  # (T, act_dim)
    acts_traces = [np.stack(tr, axis=0) for tr in layer_act_traces] if layer_act_traces else []
    return total_r, logits_arr, acts_traces

def _loss_act_l2(acts_traces):
    if acts_traces is None or len(acts_traces) == 0:
        return 0.0
    vals = []
    for A in acts_traces:  # (T,H)
        vals.append(float(np.mean(A**2)))
    return float(np.mean(vals))

def _loss_decorrelation(acts_traces, eps=1e-6):
    if acts_traces is None or len(acts_traces) == 0:
        return 0.0
    vals = []
    for A in acts_traces:  # (T,H)
        if A.shape[0] < 2 or A.shape[1] < 2:
            continue
        X = A - A.mean(axis=0, keepdims=True)  # (T,H)
        C = (X.T @ X) / (X.shape[0] + eps)     # (H,H)
        off = C - np.diag(np.diag(C))
        vals.append(float(np.mean(off**2)))
    return 0.0 if not vals else float(np.mean(vals))

def _loss_slowness(acts_traces):
    if acts_traces is None or len(acts_traces) == 0:
        return 0.0
    vals = []
    for A in acts_traces:  # (T,H)
        if A.shape[0] < 2:
            continue
        D = A[1:] - A[:-1]
        vals.append(float(np.mean(D**2)))
    return 0.0 if not vals else float(np.mean(vals))

def _loss_entropy_target(logits_arr, target):
    if logits_arr is None:
        return 0.0
    # 逐行稳定的 log-sum-exp：
    # logsumexp(x) = m + log(sum(exp(x - m))), 其中 m = max_i x_i（沿动作维）
    logits = logits_arr.astype(np.float64, copy=False)
    m = np.max(logits, axis=-1, keepdims=True)                                  # (T,1)
    logsumexp = m + np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))  # (T,1)
    logp = logits - logsumexp                                                   # (T,A)
    p = np.exp(logp)
    H = -np.sum(p * logp, axis=-1)                                              # (T,)
    H_bar = float(np.mean(H))
    return (H_bar - float(target)) ** 2

def compute_local_losses(logits_arr, acts_traces, cfg: dict):
    """返回 (loss_dict, weighted_sum)。cfg=None 时返回 0。自动屏蔽非有限值。"""
    if cfg is None:
        return {}, 0.0
    losses = {}
    total = 0.0
    # act_l2
    w = cfg.get("act_l2", {}).get("weight", 0.0)
    if w != 0.0:
        val = _loss_act_l2(acts_traces)
        if not np.isfinite(val): val = 0.0
        losses["act_l2"] = val
        total += w * val
    # decor
    w = cfg.get("decor", {}).get("weight", 0.0)
    if w != 0.0:
        val = _loss_decorrelation(acts_traces)
        if not np.isfinite(val): val = 0.0
        losses["decor"] = val
        total += w * val
    # slow
    w = cfg.get("slow", {}).get("weight", 0.0)
    if w != 0.0:
        val = _loss_slowness(acts_traces)
        if not np.isfinite(val): val = 0.0
        losses["slow"] = val
        total += w * val
    # entropy target
    ent_cfg = cfg.get("ent_target", {})
    w = ent_cfg.get("weight", 0.0)
    if w != 0.0:
        tgt = ent_cfg.get("target", 0.0)
        val = _loss_entropy_target(logits_arr, tgt)
        if not np.isfinite(val): val = 0.0
        losses["ent_target"] = val
        total += w * val
    return losses, total

@torch.no_grad()
def evaluate(env, policy, seeds=VAL_SEEDS):
    scores = []
    for sd in seeds:
        scores.append(rollout_return(env, policy, greedy=True, seed=sd))
    return float(np.mean(scores)), float(np.std(scores))

@torch.no_grad()
def rollout_return(env, policy: PolicyNet, greedy=False, reward_clip=REWARD_CLIP, seed=None):
    if seed is not None:
        set_global_seed(seed)
    obs = reset_env(env, seed=seed)
    done, total_r = False, 0.0
    while not done:
        a = policy.act_greedy(obs) if greedy else policy.act_sample(obs)
        obs, r, done = step_env(env, a)
        if reward_clip is not None:
            r = max(min(r, reward_clip), -reward_clip)
        total_r += r
    return total_r

# =========================
# 训练：Forward Learning（含 Local Loss 的黑盒扰动）
# =========================
def train_forward_local(env_train, env_val, policy: PolicyNet,
                        updates=MAX_UPDATES, eps=0.02, lr=0.03, K=K_FL,
                        local_cfg=LOCAL_LOSS_CFG, w_reward=GLOBAL_REWARD_WEIGHT):
    """
    用 Rademacher 扰动 + 黑盒梯度估计，但目标换为：
        J_total = w_reward * return - sum_i weight_i * local_loss_i
    注：local losses 在每条 rollout 内由激活/熵等“局部信号”计算，无需反传。
    local_cfg=None 时退化为纯 return 最大化（不带 local loss）。
    """
    theta = get_param_vector(policy).to(device)
    train_obj, val_means = [], []
    best_theta, best_val = theta.clone(), -1e9

    for up in range(1, updates + 1):
        g_est = torch.zeros_like(theta)
        need_local = is_cfg_enabled(local_cfg)

        # 用 K 对扰动方向估计梯度
        for k in range(K):
            delta = torch.randint_like(theta, low=0, high=2, device=device, dtype=torch.long)
            delta = delta.float().mul_(2.0).sub_(1.0)

            # θ + eps
            set_param_vector(policy, theta + eps * delta)
            R_plus, logits_plus, acts_plus = rollout_metrics(
                env_train, policy, greedy=False,
                seed=SEED + 220000 + up * 1000 + k, need_local=need_local
            )
            _, local_p = compute_local_losses(logits_plus, acts_plus, local_cfg) if need_local else ({}, 0.0)
            J_plus = w_reward * float(R_plus) - float(local_p)

            # θ - eps
            set_param_vector(policy, theta - eps * delta)
            R_minus, logits_minus, acts_minus = rollout_metrics(
                env_train, policy, greedy=False,
                seed=SEED + 230000 + up * 1000 + k, need_local=need_local
            )
            _, local_m = compute_local_losses(logits_minus, acts_minus, local_cfg) if need_local else ({}, 0.0)
            J_minus = w_reward * float(R_minus) - float(local_m)

            g_est += ((J_plus - J_minus) / (2.0 * eps)) * delta

        g_est /= max(1, K)

        # ——(1) 步长规范化/裁剪：限制每次参数步长的范数
        MAX_STEP_NORM = 1.0  # 可按需要调小，例如 0.3
        step = lr * g_est
        step_norm = torch.norm(step)
        if torch.isfinite(step_norm) and step_norm > MAX_STEP_NORM:
            step = step * (MAX_STEP_NORM / (step_norm + 1e-12))

        prev_theta = theta.clone()
        theta = theta + step
        set_param_vector(policy, theta)

        # ——(2) 若参数出现非有限值，回退并衰减学习率
        if not torch.isfinite(theta).all():
            print("[FWD][Guard] non-finite params after update; reverting and reducing lr.")
            theta = prev_theta
            set_param_vector(policy, theta)
            lr *= 0.5
            continue

        # ——(3) 评估前 logits 体检（极端数值早发现）
        with torch.no_grad():
            dummy_obs = np.zeros((policy.net[0].in_features,), dtype=np.float32)
            test_logits = policy(torch.as_tensor(dummy_obs, device=device).unsqueeze(0))
            test_logits = test_logits - torch.amax(test_logits, dim=-1, keepdim=True)
            if not torch.isfinite(test_logits).all():
                print("[FWD][Guard] non-finite logits detected; reverting and reducing lr.")
                theta = prev_theta
                set_param_vector(policy, theta)
                lr *= 0.5
                continue

        # 记录训练“目标”值（用新 θ 再 roll 一次）
        R_new, logits_new, acts_new = rollout_metrics(
            env_train, policy, greedy=False, seed=SEED + 240000 + up, need_local=need_local
        )
        _, local_new = compute_local_losses(logits_new, acts_new, local_cfg) if need_local else ({}, 0.0)
        J_new = w_reward * float(R_new) - float(local_new)
        if not np.isfinite(J_new):
            print("[FWD][Guard] non-finite objective; skipping record.")
        else:
            train_obj.append(J_new)

        # 用固定验证种子，纯“全局回报”的贪心评估
        val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
        val_means.append(val_mean)

        if val_mean > best_val:
            best_val, best_theta = val_mean, theta.clone()

        if up % EVAL_EVERY == 0 or up == 1:
            tag_local = "LocalON" if need_local else "LocalOFF"
            print(f"[FWD-{tag_local}] Update {up:3d} | train(obj): {J_new:7.2f} | VAL(greedy): {val_mean:6.1f}")

    set_param_vector(policy, best_theta)
    return train_obj, val_means, best_theta

# =========================
# 训练：REINFORCE（backprop）
# =========================
def train_backprop_reinforce(env_train, env_val, policy: PolicyNet,
                             updates=MAX_UPDATES, batch_episodes=UPD_EPISODES,
                             lr=3e-3, gamma=GAMMA, entropy_coef=0.0):
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    train_returns, val_means = [], []
    best_state, best_val = copy.deepcopy(policy.state_dict()), -1e9
    ep_counter = 0

    for up in range(1, updates + 1):
        batch_logps, batch_adv, batch_ent = [], [], []
        batch_returns = []

        for _ in range(batch_episodes):
            obs = reset_env(env_train, seed=SEED + 200000 + ep_counter)
            ep_counter += 1
            logps, entrs, rewards = [], [], []
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(obs_t)
                logits = logits - torch.amax(logits, dim=-1, keepdim=True)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        train_returns.append(float(np.mean(batch_returns)))
        with torch.no_grad():
            val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
        val_means.append(val_mean)

        if val_mean > best_val:
            best_val, best_state = val_mean, copy.deepcopy(policy.state_dict())

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[BP]   Update {up:3d} | batch_mean_return: {np.mean(batch_returns):6.1f} | VAL(greedy): {val_mean:6.1f}")

    policy.load_state_dict(best_state)
    return train_returns, val_means

# =========================
# 实验/搜索封装
# =========================
def run_one_forward_trial(base_policy, env_seed_tuple, lr_fl, eps, local_cfg):
    """返回 (score, train_curve, val_curve, best_theta, policy_state_dict)"""
    env_train = make_env(ENV_ID, seed=env_seed_tuple[0])
    env_val   = make_env(ENV_ID, seed=env_seed_tuple[1])

    policy = copy.deepcopy(base_policy).to(device)
    train_obj, val_means, best_theta = train_forward_local(
        env_train, env_val, policy,
        updates=MAX_UPDATES, eps=eps, lr=lr_fl, K=K_FL,
        local_cfg=local_cfg, w_reward=GLOBAL_REWARD_WEIGHT
    )
    k = min(STEADY_K, len(val_means))
    score = float(np.mean(val_means[-k:])) if k > 0 else float(np.mean(val_means))
    return score, train_obj, val_means, best_theta, copy.deepcopy(policy.state_dict())

def run_one_bp_trial(base_policy, env_seed_tuple, lr_bp, entropy_coef):
    """返回 (score, train_curve, val_curve, policy_state_dict)"""
    env_train = make_env(ENV_ID, seed=env_seed_tuple[0])
    env_val   = make_env(ENV_ID, seed=env_seed_tuple[1])

    policy = copy.deepcopy(base_policy).to(device)
    train_ret, val_means = train_backprop_reinforce(
        env_train, env_val, policy,
        updates=MAX_UPDATES, batch_episodes=UPD_EPISODES,
        lr=lr_bp, gamma=GAMMA, entropy_coef=entropy_coef
    )
    k = min(STEADY_K, len(val_means))
    score = float(np.mean(val_means[-k:])) if k > 0 else float(np.mean(val_means))
    return score, train_ret, val_means, copy.deepcopy(policy.state_dict())

# =========================
# 主流程：独立网格搜索 + 最优对比
# =========================
def main():
    # 统一基础网络 & 评估环境种子（trial 内部环境种子固定）
    env_seed_tuple_fwd_on  = (SEED + 10, SEED + 20)
    env_seed_tuple_fwd_off = (SEED + 12, SEED + 22)  # 与 on 略区分，避免互相污染评估
    env_seed_tuple_bp      = (SEED + 11, SEED + 21)

    # 再次设置全局种子，确保 base_policy 初始化可复现
    set_global_seed(SEED)
    # 用真实环境一次性获取维度
    tmp_env = make_env(ENV_ID, seed=SEED + 999)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n
    base_policy = PolicyNet(obs_dim, act_dim).to(device)

    # ========== Forward(Local ON) Grid Search ==========
    print("\n=== Grid Search: Forward Learning (Local Loss = ON) ===")
    fwd_on_results = []  # (lr, eps, score, final_val)
    best_fwd_on = {"score": -1e9}

    for lr_fl in FL_LR_GRID:
        for eps in FL_EPS_GRID:
            print(f"\n[FWD-ON TRY] lr={lr_fl:.5f}, eps={eps:.5f}")
            score, tr_curve, val_curve, best_theta, state_dict = run_one_forward_trial(
                base_policy, env_seed_tuple_fwd_on, lr_fl, eps, LOCAL_LOSS_CFG
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            fwd_on_results.append((lr_fl, eps, score, final_val))

            if score > best_fwd_on["score"]:
                best_fwd_on.update({
                    "score": score,
                    "lr": lr_fl,
                    "eps": eps,
                    "train_curve": tr_curve,
                    "val_curve": val_curve,
                    "best_theta": best_theta.detach().cpu(),
                    "state_dict": state_dict
                })
            print(f"[FWD-ON RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")

    print("\n[FWD-ON BEST] "
          f"lr={best_fwd_on['lr']:.5f}, eps={best_fwd_on['eps']:.5f}, "
          f"score={best_fwd_on['score']:.2f}")

    # ========== Forward(Local OFF) Grid Search ==========
    print("\n=== Grid Search: Forward Learning (Local Loss = OFF) ===")
    fwd_off_results = []  # (lr, eps, score, final_val)
    best_fwd_off = {"score": -1e9}

    for lr_fl in FL_LR_GRID:
        for eps in FL_EPS_GRID:
            print(f"\n[FWD-OFF TRY] lr={lr_fl:.5f}, eps={eps:.5f}")
            score, tr_curve, val_curve, best_theta, state_dict = run_one_forward_trial(
                base_policy, env_seed_tuple_fwd_off, lr_fl, eps, None  # 关键：local_cfg=None
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            fwd_off_results.append((lr_fl, eps, score, final_val))

            if score > best_fwd_off["score"]:
                best_fwd_off.update({
                    "score": score,
                    "lr": lr_fl,
                    "eps": eps,
                    "train_curve": tr_curve,
                    "val_curve": val_curve,
                    "best_theta": best_theta.detach().cpu(),
                    "state_dict": state_dict
                })
            print(f"[FWD-OFF RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")

    print("\n[FWD-OFF BEST] "
          f"lr={best_fwd_off['lr']:.5f}, eps={best_fwd_off['eps']:.5f}, "
          f"score={best_fwd_off['score']:.2f}")

    # ========== BP Grid Search ==========
    print("\n=== Grid Search: REINFORCE (BP) ===")
    bp_results = []  # (lr, entropy, score, final_val)
    best_bp = {"score": -1e9}

    for lr_bp in BP_LR_GRID:
        for entc in BP_ENTROPY_GRID:
            print(f"\n[BP-TRY] lr={lr_bp:.5g}, entropy_coef={entc:.1e}")
            score, tr_curve, val_curve, state_dict = run_one_bp_trial(
                base_policy, env_seed_tuple_bp, lr_bp, entc
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            bp_results.append((lr_bp, entc, score, final_val))

            if score > best_bp["score"]:
                best_bp.update({
                    "score": score,
                    "lr": lr_bp,
                    "entropy": entc,
                    "train_curve": tr_curve,
                    "val_curve": val_curve,
                    "state_dict": state_dict
                })
            print(f"[BP-RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")

    print("\n[BP-BEST] "
          f"lr={best_bp['lr']:.5g}, entropy_coef={best_bp['entropy']:.1e}, "
          f"score={best_bp['score']:.2f}")

    # ========== 最优模型最终评估与对比 ==========
    env_val_fwd_on  = make_env(ENV_ID, seed=env_seed_tuple_fwd_on[1])
    env_val_fwd_off = make_env(ENV_ID, seed=env_seed_tuple_fwd_off[1])
    env_val_bp      = make_env(ENV_ID, seed=env_seed_tuple_bp[1])

    fwd_policy_best_on = PolicyNet(obs_dim, act_dim).to(device)
    fwd_policy_best_on.load_state_dict(best_fwd_on["state_dict"])
    fwd_policy_best_off = PolicyNet(obs_dim, act_dim).to(device)
    fwd_policy_best_off.load_state_dict(best_fwd_off["state_dict"])
    bp_policy_best = PolicyNet(obs_dim, act_dim).to(device)
    bp_policy_best.load_state_dict(best_bp["state_dict"])

    final_fwd_on_mean,  final_fwd_on_std  = evaluate(env_val_fwd_on,  fwd_policy_best_on,  seeds=VAL_SEEDS)
    final_fwd_off_mean, final_fwd_off_std = evaluate(env_val_fwd_off, fwd_policy_best_off, seeds=VAL_SEEDS)
    final_bp_mean,      final_bp_std      = evaluate(env_val_bp,      bp_policy_best,      seeds=VAL_SEEDS)

    print("\n===== 最终评估（各自最优超参；贪心，固定验证种子） =====")
    print(f"  Forward(Local ON ): {final_fwd_on_mean:.2f} ± {final_fwd_on_std:.2f} "
          f"(lr={best_fwd_on['lr']:.5f},  eps={best_fwd_on['eps']:.5f})")
    print(f"  Forward(Local OFF): {final_fwd_off_mean:.2f} ± {final_fwd_off_std:.2f} "
          f"(lr={best_fwd_off['lr']:.5f}, eps={best_fwd_off['eps']:.5f})")
    print(f"  REINFORCE (BP)    : {final_bp_mean:.2f} ± {final_bp_std:.2f} "
          f"(lr={best_bp['lr']:.5g}, entropy_coef={best_bp['entropy']:.1e})")

    # ========== 绘图（最优曲线） ==========
    plt.figure(figsize=(14, 5))
    # 左：训练趋势（Forward 这里画的是 train objective；OFF 时即为回报）
    plt.subplot(1, 2, 1)
    xs_on = np.arange(1, len(best_fwd_on["train_curve"]) + 1)
    xs_off = np.arange(1, len(best_fwd_off["train_curve"]) + 1)
    xb = np.arange(1, len(best_bp["train_curve"]) + 1)
    plt.plot(xs_on,  best_fwd_on["train_curve"],  alpha=0.25, label="Forward(ON) train(obj)")
    plt.plot(xs_on,  moving_average(best_fwd_on["train_curve"]),  label="Forward(ON) EMA")
    plt.plot(xs_off, best_fwd_off["train_curve"], alpha=0.25, label="Forward(OFF) train(return)")
    plt.plot(xs_off, moving_average(best_fwd_off["train_curve"]), label="Forward(OFF) EMA")
    plt.plot(xb,     best_bp["train_curve"],      alpha=0.25, label="BP train (batch mean)")
    plt.plot(xb,     moving_average(best_bp["train_curve"]),     label="BP train EMA")
    plt.xlabel("Update #"); plt.ylabel("Objective / Return")
    plt.title("Training metrics (trend)")
    plt.legend()

    # 右：对齐 post-update 验证（贪心回报）
    plt.subplot(1, 2, 2)
    x1 = np.arange(1, len(best_fwd_on["val_curve"]) + 1)
    x2 = np.arange(1, len(best_fwd_off["val_curve"]) + 1)
    x3 = np.arange(1, len(best_bp["val_curve"]) + 1)
    plt.plot(x1, best_fwd_on["val_curve"],  label="Forward(ON) VAL (greedy)")
    plt.plot(x2, best_fwd_off["val_curve"], label="Forward(OFF) VAL (greedy)")
    plt.plot(x3, best_bp["val_curve"],      label="BP VAL (greedy)")
    plt.xlabel("Update #"); plt.ylabel("Greedy Return (fixed seeds)")
    plt.title("Aligned post-update validation (best hparams)")
    plt.legend()

    plt.tight_layout()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = f"rl_best_curves_{tag}.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()

    # ========== 保存成果 ==========
    # 模型与参数
    torch.save(best_fwd_on["best_theta"],  f"forward_on_best_theta_{tag}.pt")
    torch.save(best_fwd_on["state_dict"],  f"policy_forward_on_best_{tag}.pt")
    torch.save(best_fwd_off["best_theta"], f"forward_off_best_theta_{tag}.pt")
    torch.save(best_fwd_off["state_dict"], f"policy_forward_off_best_{tag}.pt")
    torch.save(best_bp["state_dict"],      f"policy_bp_best_{tag}.pt")

    # 搜索结果 CSV
    with open(f"forward_on_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_forward", "eps", "score_steady_avg", "final_val"])
        for row in fwd_on_results:
            w.writerow(row)
    with open(f"forward_off_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_forward", "eps", "score_steady_avg", "final_val"])
        for row in fwd_off_results:
            w.writerow(row)
    with open(f"bp_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_bp", "entropy_coef", "score_steady_avg", "final_val"])
        for row in bp_results:
            w.writerow(row)

    print("\n===== 已保存文件 =====")
    print(f"  图像: {fig_path}")
    print(f"  Forward(Local ON ): policy_forward_on_best_{tag}.pt, forward_on_best_theta_{tag}.pt, forward_on_grid_{tag}.csv")
    print(f"  Forward(Local OFF): policy_forward_off_best_{tag}.pt, forward_off_best_theta_{tag}.pt, forward_off_grid_{tag}.csv")
    print(f"  BP                : policy_bp_best_{tag}.pt, bp_grid_{tag}.csv")

if __name__ == "__main__":
    main()
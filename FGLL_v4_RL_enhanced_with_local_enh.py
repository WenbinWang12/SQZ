# fg_bp_cartpole_gridsearch_local_forward_perlayer.py
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
REWARD_CLIP = None
HIDDEN_SIZES = (128, 128)
MOVING_AVG_W = 0.95
VAL_SEEDS = list(range(90001, 90011))  # 固定验证种子
STEADY_K = 5                 # 评分时取尾部 K 次验证均值

# =========================
# Forward Learning（黑盒扰动）网格
# =========================
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
# Local Loss 配置（分层权重）
# =========================
# 说明：对于按层定义的项（act_l2/decor/slow），weight 可为：
#   - float 标量：对所有隐藏层相同；或
#   - list[float]：长度等于隐藏层数；若长度不符，会自动截断/填充最后一个权重
# 对于 ent_target（策略熵目标匹配），是全局项，不分层。
LOCAL_LOSS_CFG = {
    "act_l2":     {"weight": [0.05, 0.02]},          # 每层激活 L2 正则权重
    "decor":      {"weight": [0.02, 0.02]},          # 每层去相关权重
    "slow":       {"weight": [0.02, 0.01]},          # 每层慢变权重
    "ent_target": {"weight": 0.02, "target": 0.65}   # 全局项，不分层
}
GLOBAL_REWARD_WEIGHT = 1.0  # w_R

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
        return logits, acts  # acts: list[Tensor(batch=1, H)]

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
      - 每步 logits（用于熵）
      - 每层激活的时间序列（用于 local losses）
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
    acts_traces = [np.stack(tr, axis=0) for tr in layer_act_traces]  # each (T, H)
    return total_r, logits_arr, acts_traces

def _loss_act_l2_per_layer(acts_traces):
    """返回 per-layer 数值 list；若无激活层则返回 []"""
    if acts_traces is None or len(acts_traces) == 0:
        return []
    vals = []
    for A in acts_traces:  # (T,H)
        vals.append(float(np.mean(A**2)))
    return vals

def _loss_decorrelation_per_layer(acts_traces, eps=1e-6):
    if acts_traces is None or len(acts_traces) == 0:
        return []
    vals = []
    for A in acts_traces:  # (T,H)
        if A.shape[0] < 2 or A.shape[1] < 2:
            vals.append(0.0)
            continue
        X = A - A.mean(axis=0, keepdims=True)  # (T,H)
        C = (X.T @ X) / (X.shape[0] + eps)     # (H,H)
        off = C - np.diag(np.diag(C))
        vals.append(float(np.mean(off**2)))
    return vals

def _loss_slowness_per_layer(acts_traces):
    if acts_traces is None or len(acts_traces) == 0:
        return []
    vals = []
    for A in acts_traces:  # (T,H)
        if A.shape[0] < 2:
            vals.append(0.0)
            continue
        D = A[1:] - A[:-1]
        vals.append(float(np.mean(D**2)))
    return vals

def _loss_entropy_target_global(logits_arr, target):
    if logits_arr is None:
        return 0.0
    # 数值稳定的 entropy 计算
    x = logits_arr
    x_max = np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x - x_max)
    p = exp / np.sum(exp, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.log(p + 1e-12)
    H = -np.sum(p * logp, axis=-1)  # (T,)
    H_bar = float(np.mean(H))
    return (H_bar - float(target))**2

def _broadcast_layer_weights(weight, num_layers: int):
    """将标量或 list 的 weight 变成长度为 num_layers 的 list"""
    if num_layers <= 0:
        return []
    if isinstance(weight, (int, float)):
        return [float(weight)] * num_layers
    if isinstance(weight, (list, tuple)):
        if len(weight) == 0:
            return [0.0] * num_layers
        if len(weight) >= num_layers:
            return [float(w) for w in weight[:num_layers]]
        # 填充最后一个权重
        pad_w = float(weight[-1])
        return [float(w) for w in weight] + [pad_w] * (num_layers - len(weight))
    # 其它非法类型，按 0 处理
    return [0.0] * num_layers

def compute_local_losses(logits_arr, acts_traces, cfg: dict):
    """
    返回:
      - losses (dict): 包含逐层与全局项的原始数值，便于调试；结构：
            {
               "act_l2": [v1, v2, ...],
               "decor":  [v1, v2, ...],
               "slow":   [v1, v2, ...],
               "ent_target": v
            }
      - weighted_sum (float): 按权重加总后的局部损失总和
    """
    num_layers = 0 if acts_traces is None else len(acts_traces)
    losses = {}
    total = 0.0

    # per-layer: act_l2
    if "act_l2" in cfg:
        raw_vals = _loss_act_l2_per_layer(acts_traces) if num_layers > 0 else []
        w_list = _broadcast_layer_weights(cfg["act_l2"].get("weight", 0.0), len(raw_vals))
        losses["act_l2"] = raw_vals
        total += float(sum(w * v for w, v in zip(w_list, raw_vals)))

    # per-layer: decor
    if "decor" in cfg:
        raw_vals = _loss_decorrelation_per_layer(acts_traces) if num_layers > 0 else []
        w_list = _broadcast_layer_weights(cfg["decor"].get("weight", 0.0), len(raw_vals))
        losses["decor"] = raw_vals
        total += float(sum(w * v for w, v in zip(w_list, raw_vals)))

    # per-layer: slow
    if "slow" in cfg:
        raw_vals = _loss_slowness_per_layer(acts_traces) if num_layers > 0 else []
        w_list = _broadcast_layer_weights(cfg["slow"].get("weight", 0.0), len(raw_vals))
        losses["slow"] = raw_vals
        total += float(sum(w * v for w, v in zip(w_list, raw_vals)))

    # global: entropy target
    if "ent_target" in cfg:
        ent_w = float(cfg["ent_target"].get("weight", 0.0))
        ent_t = float(cfg["ent_target"].get("target", 0.0))
        ent_val = _loss_entropy_target_global(logits_arr, ent_t)
        losses["ent_target"] = ent_val
        total += ent_w * ent_val

    return losses, total

@torch.no_grad()
def evaluate(env, policy, seeds=VAL_SEEDS):
    scores = []
    for sd in seeds:
        scores.append(rollout_return(env, policy, greedy=True, seed=sd))
    return float(np.mean(scores)), float(np.std(scores))

@torch.no_grad()
def rollout_return(env, policy: PolicyNet, greedy=False, reward_clip=REWARD_CLIP, seed=None):
    # 保留原函数（供验证与 BP 使用）
    if seed is not None:
        set_global_seed(seed)
    obs = reset_env(env, seed=seed)
    done, total_r = False, 0.0
    while not done:
        a = policy.act_greedy(obs) if greedy else policy.act_sample(obs)
        obs, r, done = step_env(env, a)
        if reward_clip is not None:
            r = max(min(r, REWARD_CLIP), -REWARD_CLIP)
        total_r += r
    return total_r

# =========================
# 训练：Forward Learning（含分层 Local Loss 的黑盒扰动）
# =========================
def train_forward_local(env_train, env_val, policy: PolicyNet,
                        updates=MAX_UPDATES, eps=0.02, lr=0.03, K=K_FL,
                        local_cfg=LOCAL_LOSS_CFG, w_reward=GLOBAL_REWARD_WEIGHT):
    """
    用 Rademacher 扰动 + 黑盒梯度估计，但目标换为：
        J_total = w_reward * return - sum_i λ_i L_i
    其中 L_i 可能是分层的（如 act_l2/decor/slow），λ_i 为分层权重；以及全局项（熵目标）。
    """
    theta = get_param_vector(policy).to(device)
    train_obj, val_means = [], []
    best_theta, best_val = theta.clone(), -1e9

    for up in range(1, updates + 1):
        g_est = torch.zeros_like(theta)

        for k in range(K):
            # Rademacher {-1, +1}
            delta = torch.randint_like(theta, low=0, high=2, device=device, dtype=torch.long)
            delta = delta.float().mul_(2.0).sub_(1.0)

            # θ + eps
            set_param_vector(policy, theta + eps * delta)
            R_plus, logits_plus, acts_plus = rollout_metrics(
                env_train, policy, greedy=False, seed=SEED + 220000 + up * 1000 + k, need_local=True
            )
            _, local_p = compute_local_losses(logits_plus, acts_plus, local_cfg)
            J_plus = w_reward * float(R_plus) - float(local_p)

            # θ - eps
            set_param_vector(policy, theta - eps * delta)
            R_minus, logits_minus, acts_minus = rollout_metrics(
                env_train, policy, greedy=False, seed=SEED + 230000 + up * 1000 + k, need_local=True
            )
            _, local_m = compute_local_losses(logits_minus, acts_minus, local_cfg)
            J_minus = w_reward * float(R_minus) - float(local_m)

            g_est += ((J_plus - J_minus) / (2.0 * eps)) * delta

        g_est /= max(1, K)
        theta = theta + lr * g_est
        set_param_vector(policy, theta)

        # 记录训练“目标”值（用新 θ 再 roll 一次）
        R_new, logits_new, acts_new = rollout_metrics(
            env_train, policy, greedy=False, seed=SEED + 240000 + up, need_local=True
        )
        _, local_new = compute_local_losses(logits_new, acts_new, local_cfg)
        J_new = w_reward * float(R_new) - float(local_new)
        train_obj.append(J_new)

        # 验证（固定种子，贪心）
        val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
        val_means.append(val_mean)

        if val_mean > best_val:
            best_val, best_theta = val_mean, theta.clone()

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[FWD] Update {up:3d} | train(obj): {J_new:7.2f} | VAL(greedy): {val_mean:6.1f}")

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
def run_one_forward_trial(base_policy, env_seed_tuple, lr_fl, eps):
    """返回 (score, train_curve, val_curve, best_theta, policy_state_dict)"""
    env_train = make_env(ENV_ID, seed=env_seed_tuple[0])
    env_val   = make_env(ENV_ID, seed=env_seed_tuple[1])

    policy = copy.deepcopy(base_policy).to(device)
    train_obj, val_means, best_theta = train_forward_local(
        env_train, env_val, policy,
        updates=MAX_UPDATES, eps=eps, lr=lr_fl, K=K_FL,
        local_cfg=LOCAL_LOSS_CFG, w_reward=GLOBAL_REWARD_WEIGHT
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
    # 统一基础网络 & 评估环境种子（两侧 trial 内部环境种子固定）
    env_seed_tuple_fwd = (SEED + 10, SEED + 20)
    env_seed_tuple_bp  = (SEED + 11, SEED + 21)

    # 再次设置全局种子，确保 base_policy 初始化可复现
    set_global_seed(SEED)
    # 用真实环境一次性获取维度
    tmp_env = make_env(ENV_ID, seed=SEED + 999)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n
    base_policy = PolicyNet(obs_dim, act_dim).to(device)

    # ========== Forward(Local, per-layer) Grid Search ==========
    print("\n=== Grid Search: Forward Learning (with Per-Layer Local Loss) ===")
    fwd_results = []  # 保存 (lr, eps, score, final_val_mean)
    best_fwd = {"score": -1e9}

    for lr_fl in FL_LR_GRID:
        for eps in FL_EPS_GRID:
            print(f"\n[FWD-TRY] lr={lr_fl:.5f}, eps={eps:.5f}")
            score, tr_curve, val_curve, best_theta, state_dict = run_one_forward_trial(
                base_policy, env_seed_tuple_fwd, lr_fl, eps
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            fwd_results.append((lr_fl, eps, score, final_val))

            if score > best_fwd["score"]:
                best_fwd.update({
                    "score": score,
                    "lr": lr_fl,
                    "eps": eps,
                    "train_curve": tr_curve,    # train(obj)
                    "val_curve": val_curve,
                    "best_theta": best_theta.detach().cpu(),
                    "state_dict": state_dict
                })
            print(f"[FWD-RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")

    print("\n[FWD-BEST] "
          f"lr={best_fwd['lr']:.5f}, eps={best_fwd['eps']:.5f}, "
          f"score={best_fwd['score']:.2f}")

    # ========== BP Grid Search ==========
    print("\n=== Grid Search: REINFORCE (BP) ===")
    bp_results = []  # 保存 (lr, entropy, score, final_val_mean)
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
    env_val_fwd = make_env(ENV_ID, seed=env_seed_tuple_fwd[1])
    env_val_bp  = make_env(ENV_ID, seed=env_seed_tuple_bp[1])

    fwd_policy_best = PolicyNet(obs_dim, act_dim).to(device)
    fwd_policy_best.load_state_dict(best_fwd["state_dict"])
    bp_policy_best = PolicyNet(obs_dim, act_dim).to(device)
    bp_policy_best.load_state_dict(best_bp["state_dict"])

    final_fwd_mean, final_fwd_std = evaluate(env_val_fwd, fwd_policy_best, seeds=VAL_SEEDS)
    final_bp_mean,  final_bp_std  = evaluate(env_val_bp,  bp_policy_best,  seeds=VAL_SEEDS)

    print("\n===== 最终评估（各自最优超参；贪心，固定验证种子） =====")
    print(f"  Forward(Local): {final_fwd_mean:.2f} ± {final_fwd_std:.2f} "
          f"(lr={best_fwd['lr']:.5f}, eps={best_fwd['eps']:.5f})")
    print(f"  REINFORCE (BP): {final_bp_mean:.2f} ± {final_bp_std:.2f} "
          f"(lr={best_bp['lr']:.5g}, entropy_coef={best_bp['entropy']:.1e})")

    # ========== 绘图（最优曲线） ==========
    plt.figure(figsize=(12, 5))
    # 左：训练趋势（Forward 这里画的是 train objective）
    plt.subplot(1, 2, 1)
    xs = np.arange(1, len(best_fwd["train_curve"]) + 1)
    xb = np.arange(1, len(best_bp["train_curve"]) + 1)
    plt.plot(xs, best_fwd["train_curve"], alpha=0.3, label="Forward train(obj)")
    plt.plot(xs, moving_average(best_fwd["train_curve"]), label="Forward train(obj) EMA")
    plt.plot(xb, best_bp["train_curve"], alpha=0.3, label="BP train (batch mean)")
    plt.plot(xb, moving_average(best_bp["train_curve"]), label="BP train EMA")
    plt.xlabel("Update #"); plt.ylabel("Objective / Return")
    plt.title("Training metrics (trend)")
    plt.legend()

    # 右：对齐 post-update 验证（贪心回报）
    plt.subplot(1, 2, 2)
    x1 = np.arange(1, len(best_fwd["val_curve"]) + 1)
    x2 = np.arange(1, len(best_bp["val_curve"]) + 1)
    plt.plot(x1, best_fwd["val_curve"], label="Forward VAL (greedy)")
    plt.plot(x2, best_bp["val_curve"], label="BP VAL (greedy)")
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
    torch.save(best_fwd["best_theta"], f"forward_best_theta_{tag}.pt")
    torch.save(best_fwd["state_dict"], f"policy_forward_best_{tag}.pt")
    torch.save(best_bp["state_dict"],   f"policy_bp_best_{tag}.pt")

    # 搜索结果 CSV
    with open(f"forward_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_forward", "eps", "score_steady_avg", "final_val"])
        for row in fwd_results:
            w.writerow(row)
    with open(f"bp_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_bp", "entropy_coef", "score_steady_avg", "final_val"])
        for row in bp_results:
            w.writerow(row)

    print("\n===== 已保存文件 =====")
    print(f"  图像: {fig_path}")
    print(f"  Forward(Local): policy_forward_best_{tag}.pt, forward_best_theta_{tag}.pt, forward_grid_{tag}.csv")
    print(f"  BP            : policy_bp_best_{tag}.pt, bp_grid_{tag}.csv")

if __name__ == "__main__":
    main()
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
ENV_ID = "LunarLander-v2"  # 更复杂的环境
GAMMA = 0.99
MAX_UPDATES = 100            # 增加更新次数
UPD_EPISODES = 20            # 增加每次更新消耗的轨迹数
EVAL_EVERY = 5               # 打印频率
REWARD_CLIP = None
HIDDEN_SIZES = (256, 256, 128)  # 更深的隐藏层
MOVING_AVG_W = 0.95
VAL_SEEDS = list(range(90001, 90011))  # 固定验证种子
STEADY_K = 5                 # 评分时取尾部 K 次验证均值

# =========================
# “软对齐”搜索空间（可按需扩/缩）
# =========================
# —— SPSA（forward learning）
SPSA_LR_GRID = [0.0001, 0.00015, 0.0002, 0.00025, 0.0003]
SPSA_EPS_GRID = [0.05, 0.06, 0.07, 0.08, 0.09]
# K 由 UPD_EPISODES 决定（每方向两次评估，严格对齐预算）
assert UPD_EPISODES >= 2, "UPD_EPISODES 必须 >= 2"
if UPD_EPISODES % 2 != 0:
    print("[警告] 为严格对齐预算，UPD_EPISODES 应为偶数。将忽略最后 1 条轨迹。")
K_SPSA = UPD_EPISODES // 2

# —— REINFORCE（backprop）
BP_LR_GRID = [0.005, 0.01, 0.015, 0.02, 0.025]
BP_ENTROPY_GRID = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

# =========================
# 策略网络（更深的网络）
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
# 策略训练与评估（与前文一致）
# =========================
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

def train_spsa(env_train, env_val, policy: PolicyNet,
               updates=MAX_UPDATES, eps=0.02, lr=0.03, K=K_SPSA):
    theta = get_param_vector(policy).to(device)
    train_returns, val_means = [], []
    best_theta, best_val = theta.clone(), -1e9

    for up in range(1, updates + 1):
        g_est = torch.zeros_like(theta)
        for k in range(K):
            delta = torch.randint_like(theta, low=0, high=2, device=device, dtype=torch.long)
            delta = delta.float().mul_(2.0).sub_(1.0)

            set_param_vector(policy, theta + eps * delta)
            J_plus = rollout_return(env_train, policy, greedy=False,
                                    seed=SEED + 120000 + up * 1000 + k)

            set_param_vector(policy, theta - eps * delta)
            J_minus = rollout_return(env_train, policy, greedy=False,
                                     seed=SEED + 130000 + up * 1000 + k)

            g_est += ((J_plus - J_minus) / (2.0 * eps)) * delta

        g_est /= max(1, K)
        theta = theta + lr * g_est
        set_param_vector(policy, theta)

        J_new = rollout_return(env_train, policy, greedy=False, seed=SEED + 140000 + up)
        train_returns.append(J_new)

        val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
        val_means.append(val_mean)

        if val_mean > best_val:
            best_val, best_theta = val_mean, theta.clone()

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[SPSA] Update {up:3d} | train(sampled): {J_new:6.1f} | VAL(greedy): {val_mean:6.1f}")

    set_param_vector(policy, best_theta)
    return train_returns, val_means, best_theta

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


def run_one_spsa_trial(base_policy, env_seed_tuple, lr_spsa, eps):
    """返回 (score, train_curve, val_curve, best_theta, policy_state_dict)"""
    # 每个 trial 使用相同的环境种子（保证可比）
    env_train = make_env(ENV_ID, seed=env_seed_tuple[0])
    env_val   = make_env(ENV_ID, seed=env_seed_tuple[1])

    policy = copy.deepcopy(base_policy).to(device)
    train_ret, val_means, best_theta = train_spsa(
        env_train, env_val, policy,
        updates=MAX_UPDATES, eps=eps, lr=lr_spsa, K=K_SPSA
    )
    # 评分：末尾 STEADY_K 次验证均值
    k = min(STEADY_K, len(val_means))
    score = float(np.mean(val_means[-k:])) if k > 0 else float(np.mean(val_means))
    return score, train_ret, val_means, best_theta, copy.deepcopy(policy.state_dict())


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


def get_param_vector(model: nn.Module):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_param_vector(model: nn.Module, theta_vec: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(theta_vec[idx:idx+n].view_as(p))
        idx += n

# =========================
# 主流程：独立网格搜索 + 最优对比
# =========================
def main():
    # 使用相同基础网络和评估环境种子
    env_seed_tuple_spsa = (SEED + 10, SEED + 20)
    env_seed_tuple_bp   = (SEED + 11, SEED + 21)

    # 基础网络初始化
    set_global_seed(SEED)
    tmp_env = make_env(ENV_ID, seed=SEED + 999)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.n
    base_policy = PolicyNet(obs_dim, act_dim).to(device)

    # ========== SPSA Grid Search ==========
    print("\n=== Grid Search: SPSA ===")
    spsa_results = []  # 保存 (lr, eps, score, final_val_mean)
    best_spsa = {"score": -1e9}

    for lr_spsa in SPSA_LR_GRID:
        for eps in SPSA_EPS_GRID:
            print(f"\n[SPSA-TRY] lr={lr_spsa:.5f}, eps={eps:.5f}")
            score, tr_curve, val_curve, best_theta, state_dict = run_one_spsa_trial(
                base_policy, env_seed_tuple_spsa, lr_spsa, eps
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            spsa_results.append((lr_spsa, eps, score, final_val))

            if score > best_spsa["score"]:
                best_spsa.update({
                    "score": score,
                    "lr": lr_spsa,
                    "eps": eps,
                    "train_curve": tr_curve,
                    "val_curve": val_curve,
                    "best_theta": best_theta.detach().cpu(),
                    "state_dict": state_dict
                })
            print(f"[SPSA-RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")

    print("\n[SPSA-BEST] "
          f"lr={best_spsa['lr']:.5f}, eps={best_spsa['eps']:.5f}, "
          f"score={best_spsa['score']:.2f}")

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
    env_val_spsa = make_env(ENV_ID, seed=env_seed_tuple_spsa[1])
    env_val_bp   = make_env(ENV_ID, seed=env_seed_tuple_bp[1])

    spsa_policy_best = PolicyNet(obs_dim, act_dim).to(device)
    spsa_policy_best.load_state_dict(best_spsa["state_dict"])
    bp_policy_best = PolicyNet(obs_dim, act_dim).to(device)
    bp_policy_best.load_state_dict(best_bp["state_dict"])

    final_spsa_mean, final_spsa_std = evaluate(env_val_spsa, spsa_policy_best, seeds=VAL_SEEDS)
    final_bp_mean,   final_bp_std   = evaluate(env_val_bp,   bp_policy_best,   seeds=VAL_SEEDS)

    print("\n===== 最终评估（各自最优超参；贪心，固定验证种子） =====")
    print(f"  SPSA (Forward) : {final_spsa_mean:.2f} ± {final_spsa_std:.2f} "
          f"(lr={best_spsa['lr']:.5f}, eps={best_spsa['eps']:.5f})")
    print(f"  REINFORCE (BP) : {final_bp_mean:.2f} ± {final_bp_std:.2f} "
          f"(lr={best_bp['lr']:.5g}, entropy_coef={best_bp['entropy']:.1e})")

    # 绘图部分
    plt.figure(figsize=(12, 5))
    # 左：训练趋势
    plt.subplot(1, 2, 1)
    xs = np.arange(1, len(best_spsa["train_curve"]) + 1)
    xb = np.arange(1, len(best_bp["train_curve"]) + 1)
    plt.plot(xs, best_spsa["train_curve"], alpha=0.3, label="SPSA train (sampled)")
    plt.plot(xs, moving_average(best_spsa["train_curve"]), label="SPSA train EMA")
    plt.plot(xb, best_bp["train_curve"], alpha=0.3, label="BP train (batch mean)")
    plt.plot(xb, moving_average(best_bp["train_curve"]), label="BP train EMA")
    plt.xlabel("Update #"); plt.ylabel("Return")
    plt.title("Training metrics (trend)")
    plt.legend()

    # 右：对齐 post-update 验证
    plt.subplot(1, 2, 2)
    x1 = np.arange(1, len(best_spsa["val_curve"]) + 1)
    x2 = np.arange(1, len(best_bp["val_curve"]) + 1)
    plt.plot(x1, best_spsa["val_curve"], label="SPSA VAL (greedy)")
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
    torch.save(best_spsa["best_theta"], f"spsa_best_theta_{tag}.pt")
    torch.save(best_spsa["state_dict"], f"policy_spsa_best_{tag}.pt")
    torch.save(best_bp["state_dict"],   f"policy_bp_best_{tag}.pt")

    with open(f"spsa_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_spsa", "eps", "score_steady_avg", "final_val"])
        for row in spsa_results:
            w.writerow(row)
    with open(f"bp_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lr_bp", "entropy_coef", "score_steady_avg", "final_val"])
        for row in bp_results:
            w.writerow(row)

    print("\n===== 已保存文件 =====")
    print(f"  图像: {fig_path}")
    print(f"  SPSA: policy_spsa_best_{tag}.pt, spsa_best_theta_{tag}.pt, spsa_grid_{tag}.csv")
    print(f"  BP  : policy_bp_best_{tag}.pt, bp_grid_{tag}.csv")

if __name__ == "__main__":
    main()
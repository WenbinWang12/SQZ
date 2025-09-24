# -*- coding: utf-8 -*-
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
# 环境选择（更复杂场景优先）
# =========================
# 会依次尝试下列环境，若不可用则自动回退
PREFERRED_ENVS = [
    "BipedalWalker-v3",           # 更复杂，连续动作（需要 box2d）
    "LunarLanderContinuous-v2",   # 连续动作（需要 box2d）
    "LunarLander-v2",             # 回退：离散动作
]
# 若想固定环境，直接把列表改成 ["LunarLander-v2"] 等

# =========================
# 全局对齐设置 & 超参数
# =========================
GAMMA = 0.99
GAE_LAMBDA = 0.95
MAX_UPDATES = 60               # 为了较快实验，可按需加大
UPD_EPISODES = 16              # 每次更新消耗的轨迹数（REINFORCE/A2C）
EVAL_EVERY = 5
REWARD_CLIP = None
HIDDEN_SIZES = (256, 256, 128)
MOVING_AVG_W = 0.95
VAL_SEEDS = list(range(90001, 90011))
STEADY_K = 5

# =========================
# “软对齐”搜索空间
# =========================
# —— SPSA（forward learning）
SPSA_LR_GRID = [0.00015, 0.00025, 0.00035]
SPSA_EPS_GRID = [0.05, 0.08, 0.1]
assert UPD_EPISODES >= 2, "UPD_EPISODES 必须 >= 2"
if UPD_EPISODES % 2 != 0:
    print("[警告] 为严格对齐预算，UPD_EPISODES 应为偶数。将忽略最后 1 条轨迹。")
K_SPSA = UPD_EPISODES // 2

# —— REINFORCE（baseline）
BP_LR_GRID = [0.005, 0.01, 0.02]
BP_ENTROPY_GRID = [0.0001, 0.0003, 0.0005]

# —— A2C（actor/critic 分离）
A2C_ACTOR_LR_GRID = [0.0005, 0.001, 0.002]
A2C_CRITIC_LR_GRID = [0.001, 0.002]
A2C_ENTROPY_GRID = [0.0001, 0.0003]

# —— PPO（Clip）
PPO_ACTOR_LR_GRID = [0.0003, 0.0007]
PPO_CRITIC_LR_GRID = [0.001]
PPO_CLIP_GRID = [0.2, 0.25]
PPO_EPOCHS_GRID = [4, 8]
PPO_MINIBATCH_GRID = [1024]  # 按经验：LunarLander/BipedalWalker 合理 batch
PPO_ENTROPY_GRID = [0.0, 0.001]

# =========================
# 安全转换工具
# =========================
def to_numpy_safe(t: torch.Tensor):
    """避免 torch 的 .numpy() 在某些构建下报 'Numpy is not available'。
    优先返回 numpy.ndarray；若不可用则返回 Python list。"""
    t = t.detach().cpu()
    try:
        return t.numpy()
    except Exception:
        try:
            return np.asarray(t.tolist(), dtype=np.float32)
        except Exception:
            return t.tolist()

# =========================
# 通用网络模块（离散/连续统一）
# =========================
def fanin_uniform_(m: nn.Linear):
    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    if m.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(m.bias, -bound, bound)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_sizes=HIDDEN_SIZES, last_act=None):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.last_act = last_act
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            fanin_uniform_(m)

    def forward(self, x):
        out = self.net(x)
        return self.last_act(out) if self.last_act is not None else out

class PolicyHead(nn.Module):
    """
    支持：
      - Discrete: 返回 Categorical 分布（logits）
      - Continuous(Box): 返回 Diagonal Gaussian（mean, log_std）
    """
    def __init__(self, obs_dim, action_space, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.discrete = hasattr(action_space, "n")
        if self.discrete:
            self.act_dim = action_space.n
            self.body = MLP(obs_dim, hidden_sizes[-1], hidden_sizes=hidden_sizes[:-1])
            self.logits = nn.Linear(hidden_sizes[-1], self.act_dim)
            fanin_uniform_(self.logits)
        else:
            self.act_dim = action_space.shape[0]
            self.low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
            self.high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
            self.body = MLP(obs_dim, hidden_sizes[-1], hidden_sizes=hidden_sizes[:-1])
            self.mu = nn.Linear(hidden_sizes[-1], self.act_dim)
            fanin_uniform_(self.mu)
            self.log_std = nn.Parameter(torch.zeros(self.act_dim))

    def forward(self, x):
        h = self.body(x)
        if self.discrete:
            logits = self.logits(h)
            return logits, None
        else:
            mu = self.mu(h)
            log_std = self.log_std.clamp(-5, 2)  # 数值稳定
            return mu, log_std

    def dist(self, x):
        out1, out2 = self.forward(x)
        if self.discrete:
            return torch.distributions.Categorical(logits=out1)
        else:
            std = out2.exp()
            return torch.distributions.Normal(out1, std)

    @torch.no_grad()
    def act_sample(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        d = self.dist(obs_t)
        a = d.sample()
        if self.discrete:
            return int(a.item())
        else:
            a = a.squeeze(0)
            a = torch.clamp(a, self.low, self.high)
            return to_numpy_safe(a)  # 安全返回

    @torch.no_grad()
    def act_greedy(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if self.discrete:
            logits, _ = self.forward(obs_t)
            return int(torch.argmax(logits, dim=-1).item())
        else:
            mu, _ = self.forward(obs_t)
            a = torch.clamp(mu.squeeze(0), self.low, self.high)
            return to_numpy_safe(a)  # 安全返回

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.v = MLP(obs_dim, 1, hidden_sizes=hidden_sizes)
    def forward(self, x):
        return self.v(x).squeeze(-1)

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

def make_env(env_id, seed=SEED):
    env = GYM_MOD.make(env_id)
    try_seed_spaces(env, seed)
    return env

def auto_make_env(seed=SEED):
    last_err = None
    for env_id in PREFERRED_ENVS:
        try:
            env = make_env(env_id, seed)
            print(f"[Info] 使用环境: {env_id}")
            return env_id, env
        except Exception as e:
            last_err = e
            print(f"[Info] 尝试 {env_id} 失败: {e}")
    raise RuntimeError(f"没有可用环境，请检查依赖安装。最后错误: {last_err}")

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
# 策略评估
# =========================
@torch.no_grad()
def rollout_return(env, policy: PolicyHead, greedy=False, reward_clip=REWARD_CLIP, seed=None):
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

def evaluate(env, policy, seeds=VAL_SEEDS):
    scores = []
    for sd in seeds:
        scores.append(rollout_return(env, policy, greedy=True, seed=sd))
    return float(np.mean(scores)), float(np.std(scores))

# =========================
# SPSA（零阶）与参数读写
# =========================
def get_param_vector(model: nn.Module):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_param_vector(model: nn.Module, theta_vec: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(theta_vec[idx:idx+n].view_as(p))
        idx += n

def train_spsa(env_train, env_val, policy: PolicyHead,
               updates=MAX_UPDATES, eps=0.02, lr=0.03, K=K_SPSA):
    theta = get_param_vector(policy).to(device)
    train_returns, val_means = [], []
    best_theta, best_val = theta.clone(), -1e9

    for up in range(1, updates + 1):
        g_est = torch.zeros_like(theta)
        for k in range(K):
            # Rademacher 方向
            delta = torch.randint_like(theta, low=0, high=2, device=device, dtype=torch.long)
            delta = delta.float().mul_(2.0).sub_(1.0)

            # 正向/反向评估
            set_param_vector(policy, theta + eps * delta)
            J_plus = rollout_return(env_train, policy, greedy=False,
                                    seed=SEED + 120000 + up * 1000 + k)

            set_param_vector(policy, theta - eps * delta)
            J_minus = rollout_return(env_train, policy, greedy=False,
                                     seed=SEED + 130000 + up * 1000 + k)

            # —— 关键修复：把 numpy 标量显式转为 Python float（或你也可以用 torch.tensor(...)）
            scale = (float(J_plus) - float(J_minus)) / (2.0 * float(eps))
            g_est += delta * scale

        g_est /= max(1, K)
        theta = theta + lr * g_est
        set_param_vector(policy, theta)

        J_new = rollout_return(env_train, policy, greedy=False, seed=SEED + 140000 + up)
        train_returns.append(float(J_new))

        val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
        val_means.append(float(val_mean))

        if val_mean > best_val:
            best_val, best_theta = float(val_mean), theta.clone()

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[SPSA] Update {up:3d} | train(sampled): {J_new:7.2f} | VAL(greedy): {val_mean:7.2f}")

    set_param_vector(policy, best_theta)
    return train_returns, val_means, best_theta


# =========================
# REINFORCE（baseline）
# =========================
def train_reinforce(env_train, env_val, policy: PolicyHead,
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
                dist = policy.dist(obs_t)
                a = dist.sample()
                logps.append(dist.log_prob(a if policy.discrete else a).sum(-1))
                entrs.append(dist.entropy() if policy.discrete else dist.entropy().sum(-1))

                action_np = (
                    int(a.item())
                    if policy.discrete
                    else to_numpy_safe(torch.clamp(a, policy.low, policy.high).squeeze(0))
                )
                obs, r, done = step_env(env_train, action_np)
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
            print(f"[REINFORCE] Update {up:3d} | batch_mean_return: {np.mean(batch_returns):7.2f} | VAL: {val_mean:7.2f}")

    policy.load_state_dict(best_state)
    return train_returns, val_means

# =========================
# A2C（单步优势，含熵正则）
# =========================
def train_a2c(env_train, env_val, policy: PolicyHead, value_fn: ValueNet,
              updates=MAX_UPDATES, batch_episodes=UPD_EPISODES,
              actor_lr=1e-3, critic_lr=1e-3, gamma=GAMMA, entropy_coef=0.0):
    actor_opt = torch.optim.Adam(policy.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(value_fn.parameters(), lr=critic_lr)

    train_returns, val_means = [], []
    best_state = (copy.deepcopy(policy.state_dict()), copy.deepcopy(value_fn.state_dict()))
    best_val = -1e9
    ep_counter = 0

    for up in range(1, updates + 1):
        ep_returns = []

        actor_losses, critic_losses = [], []
        entropies = []

        for _ in range(batch_episodes):
            obs = reset_env(env_train, seed=SEED + 300000 + ep_counter)
            ep_counter += 1
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist = policy.dist(obs_t)
                v = value_fn(obs_t)
                a = dist.sample()
                logp = dist.log_prob(a if policy.discrete else a).sum(-1)
                entropy = dist.entropy() if policy.discrete else dist.entropy().sum(-1)

                action_np = (
                    int(a.item())
                    if policy.discrete
                    else to_numpy_safe(torch.clamp(a, policy.low, policy.high).squeeze(0))
                )
                next_obs, r, done = step_env(env_train, action_np)

                with torch.no_grad():
                    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    v_next = value_fn(next_obs_t) * (0.0 if done else 1.0)
                    td_target = torch.as_tensor([r], dtype=torch.float32, device=device) + gamma * v_next
                    adv = td_target - v

                actor_losses.append(-(logp * adv.detach()) - entropy_coef * entropy)
                critic_losses.append((v - td_target.detach()).pow(2))
                entropies.append(entropy.detach())

                obs = next_obs
            ep_returns.append(0.0)  # 占位

        # 汇总优化
        actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()
        loss = actor_loss + critic_loss

        actor_opt.zero_grad(); critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_fn.parameters()), 1.0)
        actor_opt.step(); critic_opt.step()

        # 训练指标：简单采样若干回合
        with torch.no_grad():
            tmp_returns = []
            for i in range(4):
                tmp_returns.append(rollout_return(env_train, policy, greedy=False, seed=SEED + 310000 + up*10 + i))
            train_returns.append(float(np.mean(tmp_returns)))

            val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
            val_means.append(val_mean)
            if val_mean > best_val:
                best_val = val_mean
                best_state = (copy.deepcopy(policy.state_dict()), copy.deepcopy(value_fn.state_dict()))

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[A2C] Update {up:3d} | train(sampled): {train_returns[-1]:7.2f} | VAL: {val_mean:7.2f}")

    policy.load_state_dict(best_state[0])
    value_fn.load_state_dict(best_state[1])
    return train_returns, val_means

# =========================
# PPO（Clip，GAE）
# =========================
def compute_gae(rews, vals, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t])
        nextv = vals[t+1] if t+1 < len(vals) else 0.0
        delta = rews[t] + gamma * nextv * nextnonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + np.array(vals[:T], dtype=np.float32)
    return adv, ret

def collect_trajectories(env, policy: PolicyHead, value_fn: ValueNet, steps=4096):
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
    obs = reset_env(env, seed=None)
    done = False
    while len(obs_buf) < steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = policy.dist(obs_t)
        a = dist.sample()
        logp = dist.log_prob(a if policy.discrete else a).sum(-1)
        v = value_fn(obs_t)

        action_np = (
            int(a.item())
            if policy.discrete
            else to_numpy_safe(torch.clamp(a, policy.low, policy.high).squeeze(0))
        )
        next_obs, r, done = step_env(env, action_np)

        obs_buf.append(obs)
        # 保存“未 clamp 的”动作以匹配 log_prob 的计算（PPO 里会用）
        act_buf.append(int(a.item()) if policy.discrete else to_numpy_safe(a.squeeze(0)))
        logp_buf.append(float(logp.item()))
        val_buf.append(float(v.item()))
        rew_buf.append(float(r))
        done_buf.append(done)

        obs = next_obs
        if done:
            obs = reset_env(env, seed=None)
            done = False

    return (np.array(obs_buf, dtype=np.float32),
            np.array(act_buf, dtype=np.float32),
            np.array(logp_buf, dtype=np.float32),
            np.array(val_buf, dtype=np.float32),
            np.array(rew_buf, dtype=np.float32),
            np.array(done_buf, dtype=np.bool_))

def ppo_update(policy, value_fn, optimizer_pi, optimizer_v,
               obs, act, old_logp, returns, adv,
               clip_ratio=0.2, train_pi_iters=80, train_v_iters=80, minibatch_size=1024, entropy_coef=0.0, discrete=True):
    N = len(obs)
    idxs = np.arange(N)
    adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    for _ in range(train_pi_iters):
        np.random.shuffle(idxs)
        for start in range(0, N, minibatch_size):
            mb = idxs[start:start+minibatch_size]
            obs_t = torch.as_tensor(obs[mb], dtype=torch.float32, device=device)
            act_t = torch.as_tensor(act[mb], dtype=torch.long if discrete else torch.float32, device=device)
            old_logp_t = torch.as_tensor(old_logp[mb], dtype=torch.float32, device=device)

            dist = policy.dist(obs_t)
            logp = dist.log_prob(act_t if discrete else act_t).sum(-1)
            ratio = torch.exp(logp - old_logp_t)
            ent = dist.entropy() if discrete else dist.entropy().sum(-1)

            obj1 = ratio * adv_t[mb]
            obj2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_t[mb]
            loss_pi = -(torch.min(obj1, obj2)).mean() - entropy_coef * ent.mean()

            optimizer_pi.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer_pi.step()

    for _ in range(train_v_iters):
        np.random.shuffle(idxs)
        for start in range(0, N, minibatch_size):
            mb = idxs[start:start+minibatch_size]
            obs_t = torch.as_tensor(obs[mb], dtype=torch.float32, device=device)
            ret_t = torch.as_tensor(returns[mb], dtype=torch.float32, device=device)
            v = value_fn(obs_t)
            loss_v = ((v - ret_t) ** 2).mean()
            optimizer_v.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(value_fn.parameters(), 1.0)
            optimizer_v.step()

def train_ppo(env_train, env_val, policy: PolicyHead, value_fn: ValueNet,
              updates=MAX_UPDATES, steps_per_update=4096,
              actor_lr=3e-4, critic_lr=1e-3, clip_ratio=0.2,
              pi_iters=80, v_iters=80, minibatch_size=1024, entropy_coef=0.0):

    optim_pi = torch.optim.Adam(policy.parameters(), lr=actor_lr)
    optim_v  = torch.optim.Adam(value_fn.parameters(), lr=critic_lr)

    train_returns, val_means = [], []
    best_state = (copy.deepcopy(policy.state_dict()), copy.deepcopy(value_fn.state_dict()))
    best_val = -1e9

    discrete = policy.discrete

    for up in range(1, updates + 1):
        obs, act, old_logp, vals, rews, dones = collect_trajectories(env_train, policy, value_fn, steps=steps_per_update)
        adv, rets = compute_gae(rews, vals, dones, GAMMA, GAE_LAMBDA)

        # 简单的训练期“采样回报”指标
        if len(rews) >= 200:
            n_segments = len(rews) // 200
            segment_returns = [np.sum(rews[i*200:(i+1)*200]) for i in range(n_segments)]
            train_returns.append(float(np.mean(segment_returns)))
        else:
            train_returns.append(float(np.sum(rews)))

        ppo_update(policy, value_fn, optim_pi, optim_v,
                   obs, act, old_logp, rets, adv,
                   clip_ratio=clip_ratio, train_pi_iters=pi_iters, train_v_iters=v_iters,
                   minibatch_size=minibatch_size, entropy_coef=entropy_coef, discrete=discrete)

        with torch.no_grad():
            val_mean, _ = evaluate(env_val, policy, seeds=VAL_SEEDS)
        val_means.append(val_mean)
        if val_mean > best_val:
            best_val = val_mean
            best_state = (copy.deepcopy(policy.state_dict()), copy.deepcopy(value_fn.state_dict()))

        if up % EVAL_EVERY == 0 or up == 1:
            print(f"[PPO] Update {up:3d} | VAL(greedy): {val_mean:7.2f}")

    policy.load_state_dict(best_state[0])
    value_fn.load_state_dict(best_state[1])
    return train_returns, val_means

# =========================
# Grid 试验封装
# =========================
def run_one_spsa_trial(base_policy, env_seed_tuple, lr_spsa, eps, env_id):
    env_train = make_env(env_id, seed=env_seed_tuple[0])
    env_val   = make_env(env_id, seed=env_seed_tuple[1])

    policy = copy.deepcopy(base_policy).to(device)
    train_ret, val_means, best_theta = train_spsa(
        env_train, env_val, policy,
        updates=MAX_UPDATES, eps=eps, lr=lr_spsa, K=K_SPSA
    )
    k = min(STEADY_K, len(val_means))
    score = float(np.mean(val_means[-k:])) if k > 0 else float(np.mean(val_means))
    return score, train_ret, val_means, best_theta, copy.deepcopy(policy.state_dict())

def run_one_reinforce_trial(base_policy, env_seed_tuple, lr_bp, entropy_coef, env_id):
    env_train = make_env(env_id, seed=env_seed_tuple[0])
    env_val   = make_env(env_id, seed=env_seed_tuple[1])

    policy = copy.deepcopy(base_policy).to(device)
    tr, val = train_reinforce(
        env_train, env_val, policy,
        updates=MAX_UPDATES, batch_episodes=UPD_EPISODES,
        lr=lr_bp, gamma=GAMMA, entropy_coef=entropy_coef
    )
    k = min(STEADY_K, len(val))
    score = float(np.mean(val[-k:])) if k > 0 else float(np.mean(val))
    return score, tr, val, copy.deepcopy(policy.state_dict())

def run_one_a2c_trial(base_policy, base_value, env_seed_tuple, alr, clr, entc, env_id):
    env_train = make_env(env_id, seed=env_seed_tuple[0])
    env_val   = make_env(env_id, seed=env_seed_tuple[1])

    policy = copy.deepcopy(base_policy).to(device)
    value  = copy.deepcopy(base_value).to(device)
    tr, val = train_a2c(
        env_train, env_val, policy, value,
        updates=MAX_UPDATES, batch_episodes=UPD_EPISODES,
        actor_lr=alr, critic_lr=clr, gamma=GAMMA, entropy_coef=entc
    )
    k = min(STEADY_K, len(val))
    score = float(np.mean(val[-k:])) if k > 0 else float(np.mean(val))
    return score, tr, val, (copy.deepcopy(policy.state_dict()), copy.deepcopy(value.state_dict()))

def run_one_ppo_trial(base_policy, base_value, env_seed_tuple, alr, clr, clip, epochs, mb, entc, env_id):
    env_train = make_env(env_id, seed=env_seed_tuple[0])
    env_val   = make_env(env_id, seed=env_seed_tuple[1])

    policy = copy.deepcopy(base_policy).to(device)
    value  = copy.deepcopy(base_value).to(device)
    tr, val = train_ppo(
        env_train, env_val, policy, value,
        updates=MAX_UPDATES, steps_per_update=max(2048, mb*2),
        actor_lr=alr, critic_lr=clr, clip_ratio=clip,
        pi_iters=epochs, v_iters=epochs, minibatch_size=mb, entropy_coef=entc
    )
    k = min(STEADY_K, len(val))
    score = float(np.mean(val[-k:])) if k > 0 else float(np.mean(val))
    return score, tr, val, (copy.deepcopy(policy.state_dict()), copy.deepcopy(value.state_dict()))

# =========================
# 主流程
# =========================
def main():
    # 选环境（复杂优先，失败回退）
    ENV_ID, tmp_env = auto_make_env(seed=SEED + 999)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    obs_dim = obs_space.shape[0]

    # 统一策略头部/价值网络
    base_policy = PolicyHead(obs_dim, act_space).to(device)
    base_value  = ValueNet(obs_dim).to(device)

    # 评分环境种子（保持各方法可比）
    env_seed_tuple_spsa = (SEED + 10, SEED + 20)
    env_seed_tuple_bp   = (SEED + 11, SEED + 21)
    env_seed_tuple_a2c  = (SEED + 12, SEED + 22)
    env_seed_tuple_ppo  = (SEED + 13, SEED + 23)

    # ========== SPSA Grid ==========
    print("\n=== Grid Search: SPSA ===")
    spsa_results, best_spsa = [], {"score": -1e9}
    for lr_spsa in SPSA_LR_GRID:
        for eps in SPSA_EPS_GRID:
            print(f"\n[SPSA-TRY] lr={lr_spsa:.5f}, eps={eps:.5f}")
            score, tr_curve, val_curve, best_theta, state_dict = run_one_spsa_trial(
                base_policy, env_seed_tuple_spsa, lr_spsa, eps, ENV_ID
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            spsa_results.append((lr_spsa, eps, score, final_val))
            if score > best_spsa["score"]:
                best_spsa.update({
                    "score": score, "lr": lr_spsa, "eps": eps,
                    "train_curve": tr_curve, "val_curve": val_curve,
                    "best_theta": best_theta.detach().cpu(),
                    "state_dict": state_dict
                })
            print(f"[SPSA-RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")

    print(f"\n[SPSA-BEST] lr={best_spsa['lr']:.5f}, eps={best_spsa['eps']:.5f}, score={best_spsa['score']:.2f}")

    # ========== REINFORCE Grid ==========
    print("\n=== Grid Search: REINFORCE ===")
    bp_results, best_bp = [], {"score": -1e9}
    for lr_bp in BP_LR_GRID:
        for entc in BP_ENTROPY_GRID:
            print(f"\n[REINFORCE-TRY] lr={lr_bp:.5g}, entropy_coef={entc:.1e}")
            score, tr_curve, val_curve, state_dict = run_one_reinforce_trial(
                base_policy, env_seed_tuple_bp, lr_bp, entc, ENV_ID
            )
            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
            bp_results.append((lr_bp, entc, score, final_val))
            if score > best_bp["score"]:
                best_bp.update({
                    "score": score, "lr": lr_bp, "entropy": entc,
                    "train_curve": tr_curve, "val_curve": val_curve,
                    "state_dict": state_dict
                })
            print(f"[REINFORCE-RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")
    print(f"\n[REINFORCE-BEST] lr={best_bp['lr']:.5g}, entropy_coef={best_bp['entropy']:.1e}, score={best_bp['score']:.2f}")

    # ========== A2C Grid ==========
    print("\n=== Grid Search: A2C ===")
    a2c_results, best_a2c = [], {"score": -1e9}
    for alr in A2C_ACTOR_LR_GRID:
        for clr in A2C_CRITIC_LR_GRID:
            for entc in A2C_ENTROPY_GRID:
                print(f"\n[A2C-TRY] actor_lr={alr:.4g}, critic_lr={clr:.4g}, entropy_coef={entc:.1e}")
                score, tr_curve, val_curve, state_dicts = run_one_a2c_trial(
                    base_policy, base_value, env_seed_tuple_a2c, alr, clr, entc, ENV_ID
                )
                final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
                a2c_results.append((alr, clr, entc, score, final_val))
                if score > best_a2c["score"]:
                    best_a2c.update({
                        "score": score, "alr": alr, "clr": clr, "entropy": entc,
                        "train_curve": tr_curve, "val_curve": val_curve,
                        "state_dict_pi": state_dicts[0], "state_dict_v": state_dicts[1]
                    })
                print(f"[A2C-RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")
    print(f"\n[A2C-BEST] actor_lr={best_a2c['alr']:.4g}, critic_lr={best_a2c['clr']:.4g}, entropy_coef={best_a2c['entropy']:.1e}, score={best_a2c['score']:.2f}")

    # ========== PPO Grid ==========
    print("\n=== Grid Search: PPO-Clip ===")
    ppo_results, best_ppo = [], {"score": -1e9}
    for alr in PPO_ACTOR_LR_GRID:
        for clr in PPO_CRITIC_LR_GRID:
            for clip in PPO_CLIP_GRID:
                for epc in PPO_EPOCHS_GRID:
                    for mb in PPO_MINIBATCH_GRID:
                        for entc in PPO_ENTROPY_GRID:
                            print(f"\n[PPO-TRY] actor_lr={alr:.4g}, critic_lr={clr:.4g}, clip={clip:.2f}, epochs={epc}, mb={mb}, entropy={entc:.1e}")
                            score, tr_curve, val_curve, state_dicts = run_one_ppo_trial(
                                base_policy, base_value, env_seed_tuple_ppo, alr, clr, clip, epc, mb, entc, ENV_ID
                            )
                            final_val = float(val_curve[-1]) if len(val_curve) > 0 else float("nan")
                            ppo_results.append((alr, clr, clip, epc, mb, entc, score, final_val))
                            if score > best_ppo["score"]:
                                best_ppo.update({
                                    "score": score, "alr": alr, "clr": clr, "clip": clip, "epochs": epc, "mb": mb, "entropy": entc,
                                    "train_curve": tr_curve, "val_curve": val_curve,
                                    "state_dict_pi": state_dicts[0], "state_dict_v": state_dicts[1]
                                })
                            print(f"[PPO-RESULT] score(steady-avg)={score:.2f}, final_val={final_val:.2f}")
    print(f"\n[PPO-BEST] actor_lr={best_ppo['alr']:.4g}, critic_lr={best_ppo['clr']:.4g}, clip={best_ppo['clip']:.2f}, "
          f"epochs={best_ppo['epochs']}, mb={best_ppo['mb']}, entropy={best_ppo['entropy']:.1e}, score={best_ppo['score']:.2f}")

    # ========== 最优模型最终评估与对比 ==========
    def rebuild_env(seed_val):  # 避免复用旧 env 的隐藏状态
        return make_env(ENV_ID, seed=seed_val)

    env_val_spsa = rebuild_env(SEED + 20)
    env_val_bp   = rebuild_env(SEED + 21)
    env_val_a2c  = rebuild_env(SEED + 22)
    env_val_ppo  = rebuild_env(SEED + 23)

    # 恢复最优权重
    spsa_policy_best = PolicyHead(obs_dim, act_space).to(device)
    spsa_policy_best.load_state_dict(best_spsa["state_dict"])
    reinforce_best = PolicyHead(obs_dim, act_space).to(device)
    reinforce_best.load_state_dict(best_bp["state_dict"])
    a2c_pi_best = PolicyHead(obs_dim, act_space).to(device)
    a2c_v_best  = ValueNet(obs_dim).to(device)
    a2c_pi_best.load_state_dict(best_a2c["state_dict_pi"])
    a2c_v_best.load_state_dict(best_a2c["state_dict_v"])
    ppo_pi_best = PolicyHead(obs_dim, act_space).to(device)
    ppo_v_best  = ValueNet(obs_dim).to(device)
    ppo_pi_best.load_state_dict(best_ppo["state_dict_pi"])
    ppo_v_best.load_state_dict(best_ppo["state_dict_v"])

    final_spsa_mean, final_spsa_std = evaluate(env_val_spsa, spsa_policy_best, seeds=VAL_SEEDS)
    final_bp_mean,   final_bp_std   = evaluate(env_val_bp,   reinforce_best,   seeds=VAL_SEEDS)
    final_a2c_mean,  final_a2c_std  = evaluate(env_val_a2c,  a2c_pi_best,      seeds=VAL_SEEDS)
    final_ppo_mean,  final_ppo_std  = evaluate(env_val_ppo,  ppo_pi_best,      seeds=VAL_SEEDS)

    print("\n===== 最终评估（各自最优超参；贪心，固定验证种子） =====")
    print(f"  SPSA (Forward) : {final_spsa_mean:.2f} ± {final_spsa_std:.2f} (lr={best_spsa['lr']:.5f}, eps={best_spsa['eps']:.5f})")
    print(f"  REINFORCE (BP) : {final_bp_mean:.2f} ± {final_bp_std:.2f} (lr={best_bp['lr']:.5g}, entropy={best_bp['entropy']:.1e})")
    print(f"  A2C (BP)       : {final_a2c_mean:.2f} ± {final_a2c_std:.2f} (actor_lr={best_a2c['alr']:.4g}, critic_lr={best_a2c['clr']:.4g}, entropy={best_a2c['entropy']:.1e})")
    print(f"  PPO-Clip (BP)  : {final_ppo_mean:.2f} ± {final_ppo_std:.2f} (actor_lr={best_ppo['alr']:.4g}, critic_lr={best_ppo['clr']:.4g}, clip={best_ppo['clip']:.2f}, epochs={best_ppo['epochs']}, mb={best_ppo['mb']}, entropy={best_ppo['entropy']:.1e})")

    # 绘图
    plt.figure(figsize=(13, 5))
    # 左：训练趋势
    plt.subplot(1, 2, 1)
    def plot_curve(name, curve, ema=True):
        x = np.arange(1, len(curve)+1)
        plt.plot(x, curve, alpha=0.25, label=f"{name} train")
        if ema:
            plt.plot(x, moving_average(curve), label=f"{name} train EMA")
    plot_curve("SPSA", best_spsa["train_curve"])
    plot_curve("REINFORCE", best_bp["train_curve"])
    plot_curve("A2C", best_a2c["train_curve"])
    plot_curve("PPO", best_ppo["train_curve"])
    plt.xlabel("Update #"); plt.ylabel("Return"); plt.title("Training metrics (trend)")
    plt.legend()

    # 右：对齐 post-update 验证
    plt.subplot(1, 2, 2)
    def plot_val(name, curve):
        x = np.arange(1, len(curve)+1); plt.plot(x, curve, label=f"{name} VAL (greedy)")
    plot_val("SPSA", best_spsa["val_curve"])
    plot_val("REINFORCE", best_bp["val_curve"])
    plot_val("A2C", best_a2c["val_curve"])
    plot_val("PPO", best_ppo["val_curve"])
    plt.xlabel("Update #"); plt.ylabel("Greedy Return (fixed seeds)")
    plt.title("Aligned post-update validation (best hparams)")
    plt.legend()

    plt.tight_layout()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = f"rl_best_curves_{tag}.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()

    # 保存成果
    torch.save(best_spsa["best_theta"], f"spsa_best_theta_{tag}.pt")
    torch.save(best_spsa["state_dict"], f"policy_spsa_best_{tag}.pt")
    torch.save(best_bp["state_dict"],   f"policy_reinforce_best_{tag}.pt")
    torch.save(best_a2c["state_dict_pi"], f"policy_a2c_actor_best_{tag}.pt")
    torch.save(best_a2c["state_dict_v"],  f"value_a2c_best_{tag}.pt")
    torch.save(best_ppo["state_dict_pi"], f"policy_ppo_actor_best_{tag}.pt")
    torch.save(best_ppo["state_dict_v"],  f"value_ppo_best_{tag}.pt")

    with open(f"spsa_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lr_spsa", "eps", "score_steady_avg", "final_val"])
        for row in spsa_results: w.writerow(row)
    with open(f"reinforce_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lr", "entropy_coef", "score_steady_avg", "final_val"])
        for row in bp_results: w.writerow(row)
    with open(f"a2c_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["actor_lr", "critic_lr", "entropy_coef", "score_steady_avg", "final_val"])
        for row in a2c_results: w.writerow(row)
    with open(f"ppo_grid_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["actor_lr", "critic_lr", "clip", "epochs", "minibatch", "entropy_coef", "score_steady_avg", "final_val"])
        for row in ppo_results: w.writerow(row)

    print("\n===== 已保存文件 =====")
    print(f"  图像: {fig_path}")
    print(f"  SPSA: policy_spsa_best_{tag}.pt, spsa_best_theta_{tag}.pt, spsa_grid_{tag}.csv")
    print(f"  REINFORCE: policy_reinforce_best_{tag}.pt, reinforce_grid_{tag}.csv")
    print(f"  A2C: policy_a2c_actor_best_{tag}.pt, value_a2c_best_{tag}.pt, a2c_grid_{tag}.csv")
    print(f"  PPO: policy_ppo_actor_best_{tag}.pt, value_ppo_best_{tag}.pt, ppo_grid_{tag}.csv")

if __name__ == "__main__":
    main()

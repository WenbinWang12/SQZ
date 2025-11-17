"""
maze_compare_arch_optim_all.py

在走迷宫任务上对比多种网络架构 (CNN / Transformer / Mambo-style / Linear-Attention) +
多种优化器（含 Muon / Lion / Shampoo / Adam / AdamW / Muon+AdamW），
在多种学习率下比较时间与精度，并自动生成可视化图表和汇总日志。
"""

#############################################
#                  Setup                    #
#############################################

import os
import sys
import uuid
import random
import time
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True

# 保存当前脚本代码，便于结果复现
with open(sys.argv[0]) as f:
    code = f.read()

#############################################
#               Muon optimizer              #
#############################################

def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Muon 的零阶幂正交化迭代（不使用 torch.compile，避免 AUTOTUNE 日志）。
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(Optimizer):
    """
    对 4D filter 做归一化 + 零阶幂白化的 Muon 优化器。
    """
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf

                # normalize weight
                p.data.mul_(len(p.data) ** 0.5 / (p.data.norm() + 1e-12))
                # whiten update
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)
                p.data.add_(update, alpha=-lr)
        return None

#############################################
#              Lion optimizer               #
#############################################

class Lion(Optimizer):
    """
    Lion optimizer（简化版）。
    update = sign( beta1 * m + (1 - beta1) * g )
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 value")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2 value")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if weight_decay != 0.0:
                    g = g.add(p.data, alpha=weight_decay)

                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                m = state['exp_avg']

                # m_t = beta2 * m + (1 - beta2) * g
                m.mul_(beta2).add_(g, alpha=1 - beta2)
                # u_t = beta1 * m_t + (1 - beta1) * g
                u = beta1 * m + (1 - beta1) * g

                p.add_(u.sign(), alpha=-lr)
        return None

#############################################
#            Simplified Shampoo             #
#############################################

class Shampoo(Optimizer):
    """
    非完整、简化版 layer-wise Shampoo：
    - 只对 ndim>=2 且较小的矩阵做 Shampoo，其余退化为 SGD。
    """
    def __init__(self, params, lr=1e-2, eps=1e-4, weight_decay=0.0, max_dim=512):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay, max_dim=max_dim)
        super().__init__(params, defaults)

    def _matrix_power(self, mat, exponent):
        e, Q = torch.linalg.eigh(mat)
        e = torch.clamp(e, min=1e-12)
        e_pow = e.pow(exponent)
        return (Q * e_pow.unsqueeze(0)) @ Q.transpose(-1, -2)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            max_dim = group['max_dim']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if weight_decay != 0.0:
                    g = g.add(p.data, alpha=weight_decay)

                state = self.state[p]

                if p.ndim >= 2:
                    d0 = p.shape[0]
                    d1 = p.numel() // d0
                    if d1 > max_dim:
                        # 太大的矩阵退化为 SGD
                        p.add_(g, alpha=-lr)
                        continue

                    g2d = g.reshape(d0, d1).float()

                    if 'G' not in state:
                        state['G'] = torch.eye(d0, device=g2d.device, dtype=torch.float32) * eps
                        state['H'] = torch.eye(d1, device=g2d.device, dtype=torch.float32) * eps

                    G = state['G']
                    H = state['H']

                    G.add_(g2d @ g2d.t())
                    H.add_(g2d.t() @ g2d)

                    G_inv_root = self._matrix_power(G + eps * torch.eye(d0, device=G.device), -0.25)
                    H_inv_root = self._matrix_power(H + eps * torch.eye(d1, device=H.device), -0.25)

                    pre_g = G_inv_root @ g2d @ H_inv_root
                    pre_g = pre_g.reshape_as(p).to(p.dtype)
                    p.add_(pre_g, alpha=-lr)
                else:
                    # 标量/向量参数用 SGD
                    p.add_(g, alpha=-lr)
        return None

#############################################
#              Maze task utilities          #
#############################################

# 方向编码：0=上, 1=下, 2=左, 3=右
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def encode_action(p0, p1):
    r0, c0 = p0
    r1, c1 = p1
    dr = r1 - r0
    dc = c1 - c0
    for a, (adr, adc) in enumerate(DIRS):
        if dr == adr and dc == adc:
            return a
    raise ValueError(f"Invalid move from {p0} to {p1}")

def bfs_shortest_path(grid, start, goal):
    """
    在 grid 上做 BFS，返回 start->goal 的最短路径（包含两端）。
    grid: torch.bool, True=墙, False=空
    start, goal: (r,c)
    若不可达返回 None。
    """
    H, W = grid.shape
    sr, sc = start
    gr, gc = goal
    if grid[sr, sc] or grid[gr, gc]:
        return None

    q = deque()
    q.append((sr, sc))
    visited = [[False] * W for _ in range(H)]
    visited[sr][sc] = True
    parent = {}

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            # 回溯路径
            path = [(r, c)]
            while (r, c) != (sr, sc):
                r, c = parent[(r, c)]
                path.append((r, c))
            path.reverse()
            return path
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not grid[nr, nc] and not visited[nr][nc]:
                visited[nr][nc] = True
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return None

def generate_random_maze(H=15, W=15, wall_prob=0.2, max_tries=200):
    """
    随机生成一个 15x15 迷宫：
    - 随机布置墙
    - 随机选择起点和终点
    - 使用 BFS 确保有可行路径
    返回: (grid, start, goal, path)
    """
    for _ in range(max_tries):
        grid = (torch.rand(H, W) < wall_prob)
        # 保证边缘少一点墙，防止太难走
        grid[0, :] = False
        grid[-1, :] = False
        grid[:, 0] = False
        grid[:, -1] = False

        free = (~grid).nonzero(as_tuple=False)
        if free.size(0) < 2:
            continue
        idx = torch.randperm(free.size(0))[:2]
        sr, sc = free[idx[0]].tolist()
        gr, gc = free[idx[1]].tolist()
        start = (sr, sc)
        goal = (gr, gc)

        if start == goal:
            continue

        path = bfs_shortest_path(grid, start, goal)
        if path is None or len(path) < 2:
            continue
        return grid, start, goal, path
    raise RuntimeError("Failed to generate a valid maze after many tries")

def make_obs(grid, agent, goal):
    """
    根据 grid + agent 位置 + goal 位置构造 3 通道观测：
    ch0: 墙 (1=墙, 0=空)
    ch1: agent one-hot
    ch2: goal one-hot
    返回: (3,H,W)
    """
    H, W = grid.shape
    obs = torch.zeros(3, H, W, dtype=torch.float32)
    obs[0] = grid.float()
    ar, ac = agent
    gr, gc = goal
    obs[1, ar, ac] = 1.0
    obs[2, gr, gc] = 1.0
    return obs

class MazeDataset(Dataset):
    """
    走迷宫监督数据：
    - 每个样本是一个迷宫 + 起点 + 终点
    - label 是起点处最优第一步动作（上/下/左/右）
    - 训练时做动作分类，评估时用 greedy policy rollout 统计成功率
    """
    def __init__(self, num_samples=1000, H=15, W=15, wall_prob=0.2):
        super().__init__()
        self.H = H
        self.W = W
        obs_list = []
        action_list = []
        maps = []
        starts = []
        goals = []

        for _ in range(num_samples):
            grid, start, goal, path = generate_random_maze(H=H, W=W, wall_prob=wall_prob)
            first_action = encode_action(path[0], path[1])
            obs = make_obs(grid, start, goal)
            obs_list.append(obs)
            action_list.append(first_action)
            maps.append(grid)
            starts.append(torch.tensor(start, dtype=torch.long))
            goals.append(torch.tensor(goal, dtype=torch.long))

        self.obs = torch.stack(obs_list)                 # (N,3,H,W)
        self.actions = torch.tensor(action_list)         # (N,)
        self.maps = torch.stack(maps)                    # (N,H,W) bool
        self.starts = torch.stack(starts)                # (N,2)
        self.goals = torch.stack(goals)                  # (N,2)

    def __len__(self):
        return self.obs.size(0)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

#############################################
#             Maze networks (CNN)           #
#############################################

class MazeNetSmall(nn.Module):
    """
    浅层 CNN：输入 (B,3,15,15)，输出 (B,4) 动作 logits
    """
    def __init__(self, in_ch=3, hidden=64, num_actions=4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, num_actions, bias=False)

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d,
                              nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.Linear, nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()

    def forward(self, x):
        # x: (B,3,H,W)
        x = self.body(x)              # (B,C,H,W)
        x = x.mean(dim=(2, 3))        # global average pooling -> (B,C)
        x = self.head(x)              # (B,4)
        return x

class MazeNetDeep(nn.Module):
    """
    深层 CNN：更多 conv 层、更宽通道。
    """
    def __init__(self, in_ch=3, hidden=128, depth=6, num_actions=4):
        super().__init__()
        layers = []
        c_in = in_ch
        for _ in range(depth):
            layers.append(nn.Conv2d(c_in, hidden, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden))
            layers.append(nn.GELU())
            c_in = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, num_actions, bias=False)

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d,
                              nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.Linear, nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()

    def forward(self, x):
        x = self.body(x)
        x = x.mean(dim=(2, 3))
        x = self.head(x)
        return x

class ResBlock(nn.Module):
    """
    残差块：Conv-BN-GELU-Conv-BN + skip。
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        return out

class MazeNetRes(nn.Module):
    """
    残差 CNN：conv_in + 多个 ResBlock + global pooling。
    """
    def __init__(self, in_ch=3, channels=96, num_blocks=3, num_actions=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(channels))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, num_actions, bias=False)

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d,
                              nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.Linear, nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.act(x)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        x = self.head(x)
        return x

#############################################
#   Transformer-based Maze networks         #
#############################################

class MazeNetTransSmall(nn.Module):
    """
    浅层 Transformer：
    - 把 15x15 的每个格子当成一个 token，通道为 (墙, agent, goal)
    - d_model=128, nhead=4, num_layers=2
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, num_actions=4, H=15, W=15):
        super().__init__()
        self.H = H
        self.W = W
        self.in_proj = nn.Linear(3, d_model)
        self.pos_embed = nn.Embedding(H * W, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_actions)

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d,
                              nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.Linear, nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()

    def forward(self, x):
        # x: (B,3,H,W)
        B, C, H, W = x.shape
        assert H == self.H and W == self.W
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B,L,3)
        pos = torch.arange(H * W, device=x.device).unsqueeze(0).expand(B, -1)  # (B,L)
        x = self.in_proj(x) + self.pos_embed(pos)  # (B,L,D)
        x = self.encoder(x)                        # (B,L,D)
        x = x.mean(dim=1)                          # global average over tokens
        x = self.head(x)                           # (B,4)
        return x

class MazeNetTransDeep(nn.Module):
    """
    深层 Transformer：
    - d_model=192, nhead=6, num_layers=6
    """
    def __init__(self, d_model=192, nhead=6, num_layers=6, num_actions=4, H=15, W=15):
        super().__init__()
        self.H = H
        self.W = W
        self.in_proj = nn.Linear(3, d_model)
        self.pos_embed = nn.Embedding(H * W, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_actions)

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d,
                              nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.Linear, nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.H and W == self.W
        x = x.view(B, C, H * W).permute(0, 2, 1)
        pos = torch.arange(H * W, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.in_proj(x) + self.pos_embed(pos)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

#############################################
#       Mambo-style (SSM-like) networks     #
#############################################

class MamboBlock(nn.Module):
    """
    简化版 Mambo-style block：
    - LN
    - 线性投影 -> (u, v) 分支
    - v 做 depthwise conv（1D over sequence）
    - 点乘 u * v，再线性投影回 d_model
    """
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_ff)
        self.conv = nn.Conv1d(d_ff, d_ff, kernel_size=3, padding=1, groups=d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B,L,D)
        residual = x
        x = self.norm(x)
        u, v = self.in_proj(x).chunk(2, dim=-1)      # (B,L,d_ff)
        v = self.act(v)
        v = v.transpose(1, 2)                        # (B,d_ff,L)
        v = self.conv(v)                             # depthwise conv
        v = v.transpose(1, 2)                        # (B,L,d_ff)
        x = u * v
        x = self.out_proj(x)                         # (B,L,D)
        return residual + x

class MazeNetMamboSmall(nn.Module):
    """
    浅层 Mambo-style 序列模型：
    - d_model=128, num_blocks=3
    """
    def __init__(self, d_model=128, num_blocks=3, num_actions=4, H=15, W=15):
        super().__init__()
        self.H = H
        self.W = W
        self.in_proj = nn.Linear(3, d_model)
        self.pos_embed = nn.Embedding(H * W, d_model)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(MamboBlock(d_model))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(d_model, num_actions)

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d,
                              nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.Linear, nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.H and W == self.W
        x = x.view(B, C, H * W).permute(0, 2, 1)      # (B,L,3)
        pos = torch.arange(H * W, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.in_proj(x) + self.pos_embed(pos)     # (B,L,D)
        x = self.blocks(x)                            # (B,L,D)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

class MazeNetMamboDeep(nn.Module):
    """
    深层 Mambo-style 序列模型：
    - d_model=192, num_blocks=6
    """
    def __init__(self, d_model=192, num_blocks=6, num_actions=4, H=15, W=15):
        super().__init__()
        self.H = H
        self.W = W
        self.in_proj = nn.Linear(3, d_model)
        self.pos_embed = nn.Embedding(H * W, d_model)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(MamboBlock(d_model))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(d_model, num_actions)

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d,
                              nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.Linear, nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.H and W == self.W
        x = x.view(B, C, H * W).permute(0, 2, 1)
        pos = torch.arange(H * W, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.in_proj(x) + self.pos_embed(pos)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

#############################################
#       Linear Attention-based network      #
#############################################

class LinearAttentionLayer(nn.Module):
    """
    线性注意力 Encoder layer：
    - LN
    - multi-head linear attention (φ(x)=ELU(x)+1)
    - 残差 + FFN
    """
    def __init__(self, d_model=128, nhead=4, d_ff=None):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if d_ff is None:
            d_ff = 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.act_phi = nn.ELU()

    def _linear_attention(self, q, k, v, eps=1e-6):
        """
        q,k,v: (B,H,L,Dh)
        线性注意力近似：
        out_t = φ(q_t) (∑_s φ(k_s) v_s^T) / (φ(q_t) ∑_s φ(k_s))
        """
        B, H, L, Dh = q.shape
        phi_q = self.act_phi(q) + 1.0          # (B,H,L,Dh)
        phi_k = self.act_phi(k) + 1.0          # (B,H,L,Dh)

        # KV: (B,H,Dh,Dh)
        kv = torch.einsum('bhld,bhlv->bhdv', phi_k, v)
        # k_sum: (B,H,Dh)
        k_sum = phi_k.sum(dim=-2)

        # out: (B,H,L,Dh)
        out = torch.einsum('bhld,bhdv->bhlv', phi_q, kv)
        # denom: (B,H,L)
        denom = torch.einsum('bhld,bhd->bhl', phi_q, k_sum) + eps
        out = out / denom.unsqueeze(-1)
        return out

    def forward(self, x):
        # x: (B,L,D)
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)  # (B,L,3D)
        q, k, v = qkv.chunk(3, dim=-1)
        B, L, D = q.shape
        H = self.nhead
        Dh = self.head_dim

        # (B,H,L,Dh)
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        attn = self._linear_attention(q, k, v)           # (B,H,L,Dh)
        attn = attn.transpose(1, 2).reshape(B, L, D)     # (B,L,D)
        x = residual + self.out_proj(attn)

        # FFN
        residual2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual2 + x
        return x

class MazeNetLinAttnSmall(nn.Module):
    """
    基于 Linear Attention 的浅层 Transformer 风格模型：
    - d_model=128, nhead=4, num_layers=4
    """
    def __init__(self, d_model=128, nhead=4, num_layers=4, num_actions=4, H=15, W=15):
        super().__init__()
        self.H = H
        self.W = W
        self.in_proj = nn.Linear(3, d_model)
        self.pos_embed = nn.Embedding(H * W, d_model)
        layers = []
        for _ in range(num_layers):
            layers.append(LinearAttentionLayer(d_model=d_model, nhead=nhead))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, num_actions)

    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d,
                              nn.BatchNorm2d, nn.BatchNorm1d,
                              nn.Linear, nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.H and W == self.W
        x = x.view(B, C, H * W).permute(0, 2, 1)      # (B,L,3)
        pos = torch.arange(H * W, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.in_proj(x) + self.pos_embed(pos)     # (B,L,D)
        x = self.layers(x)                            # (B,L,D)
        x = x.mean(dim=1)
        x = self.head(x)
        return x

#############################################
#           Model factory (build_model)     #
#############################################

def build_model(arch_name: str) -> nn.Module:
    if arch_name == "small":
        return MazeNetSmall()
    elif arch_name == "deep":
        return MazeNetDeep()
    elif arch_name == "res":
        return MazeNetRes()
    elif arch_name == "trans_small":
        return MazeNetTransSmall()
    elif arch_name == "trans_deep":
        return MazeNetTransDeep()
    elif arch_name == "mambo_small":
        return MazeNetMamboSmall()
    elif arch_name == "mambo_deep":
        return MazeNetMamboDeep()
    elif arch_name == "linattn_small":
        return MazeNetLinAttnSmall()
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-' * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-' * len(print_string))

logging_columns_list = ['run                         ', 'epoch', 'train_acc', 'val_acc', 'maze_success', 'time_seconds']

def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        key = col.strip()
        var = variables.get(key, None)
        if isinstance(var, (int, str)):
            res = str(var)
        elif isinstance(var, float):
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#               Evaluation                 #
############################################

def evaluate_acc(model, loader, device):
    """
    迷宫第一步动作分类准确率。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for obs, action in loader:
            obs = obs.to(device)
            action = action.to(device)
            logits = model(obs)              # (B,4)
            pred = logits.argmax(1)
            correct += (pred == action).sum().item()
            total += action.numel()
    return correct / total

def step_in_maze(pos, action, grid):
    """
    在迷宫中执行一步动作：
    - 如果撞墙或越界，就原地不动。
    """
    r, c = pos
    dr, dc = DIRS[action]
    nr, nc = r + dr, c + dc
    H, W = grid.shape
    if 0 <= nr < H and 0 <= nc < W and not grid[nr, nc]:
        return (nr, nc)
    else:
        return (r, c)

def evaluate_maze_success(model, dataset, device, max_steps_factor=4):
    """
    使用 greedy policy 从起点出发，按网络输出的 argmax 动作滚动，
    如果在步数上限内到达终点则记为成功。
    返回成功比例。
    """
    model.eval()
    H = dataset.H
    W = dataset.W
    max_steps = H * W * max_steps_factor
    successes = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            grid = dataset.maps[i].to(device)      # (H,W) bool
            start = dataset.starts[i].tolist()
            goal = dataset.goals[i].tolist()
            pos = tuple(start)
            for _ in range(max_steps):
                obs = make_obs(grid.cpu(), pos, goal).unsqueeze(0).to(device)
                logits = model(obs)                # (1,4)
                action = logits.argmax(1).item()
                pos = step_in_maze(pos, action, grid)
                if pos == tuple(goal):
                    successes += 1
                    break
    return successes / len(dataset)

############################################
#            Optimizer factory             #
############################################

def make_optimizers(model, optimizer_name, lr, wd_effective=1e-4):
    """
    构造不同优化器：
    - 'sgd'        ：SGD + momentum + Nesterov，所有参数。
    - 'sgd_no_mom'：纯 SGD。
    - 'adam'      ：Adam。
    - 'adamw'     ：AdamW。
    - 'rmsprop'   ：RMSprop。
    - 'lion'      ：Lion。
    - 'muon'      ：Conv filter 用 Muon，其余 用 SGD。
    - 'muon_adamw': Conv filter 用 Muon，其余 用 AdamW。
    - 'shampoo'   ：简化版 Shampoo。
    """
    params_all = [p for p in model.parameters() if p.requires_grad]
    filter_params = [p for p in params_all if p.ndim == 4]
    other_params = [p for p in params_all if p.ndim != 4]

    optimizers = []
    desc = ""

    if optimizer_name == "sgd":
        opt = torch.optim.SGD(params_all, lr=lr, momentum=0.9, nesterov=True,
                              weight_decay=wd_effective)
        optimizers = [opt]
        desc = f"SGD(momentum=0.9, nesterov=True, lr={lr})"

    elif optimizer_name == "sgd_no_mom":
        opt = torch.optim.SGD(params_all, lr=lr, momentum=0.0, nesterov=False,
                              weight_decay=wd_effective)
        optimizers = [opt]
        desc = f"SGD(momentum=0.0, lr={lr})"

    elif optimizer_name == "adam":
        opt = torch.optim.Adam(params_all, lr=lr, weight_decay=wd_effective)
        optimizers = [opt]
        desc = f"Adam(lr={lr}, weight_decay={wd_effective})"

    elif optimizer_name == "adamw":
        opt = torch.optim.AdamW(params_all, lr=lr, weight_decay=wd_effective)
        optimizers = [opt]
        desc = f"AdamW(lr={lr}, weight_decay={wd_effective})"

    elif optimizer_name == "rmsprop":
        opt = torch.optim.RMSprop(params_all, lr=lr, alpha=0.99, momentum=0.9,
                                  weight_decay=wd_effective)
        optimizers = [opt]
        desc = f"RMSprop(lr={lr}, momentum=0.9, alpha=0.99)"

    elif optimizer_name == "lion":
        opt = Lion(params_all, lr=lr, betas=(0.9, 0.99), weight_decay=wd_effective)
        optimizers = [opt]
        desc = f"Lion(lr={lr}, betas=(0.9,0.99))"

    elif optimizer_name == "muon":
        # Conv filter: Muon；其他：SGD
        muon_lr = lr * 4.0  # 给 Muon 略大步长
        opt_sgd = torch.optim.SGD(other_params, lr=lr, momentum=0.9, nesterov=True,
                                  weight_decay=wd_effective)
        opt_muon = Muon(filter_params, lr=muon_lr, momentum=0.6, nesterov=True)
        optimizers = [opt_sgd, opt_muon]
        desc = f"Muon(conv filters, lr={muon_lr}) + SGD(others, lr={lr})"

    elif optimizer_name == "muon_adamw":
        # Conv filter: Muon；其他：AdamW
        muon_lr = lr * 4.0
        if len(filter_params) == 0:
            # 没有 Conv filter，就退化成纯 AdamW
            opt = torch.optim.AdamW(params_all, lr=lr, weight_decay=wd_effective)
            optimizers = [opt]
            desc = (f"AdamW(lr={lr}, weight_decay={wd_effective}) "
                    f"(no conv filters, Muon unused)")
        else:
            opt_adamw = torch.optim.AdamW(other_params, lr=lr, weight_decay=wd_effective)
            opt_muon = Muon(filter_params, lr=muon_lr, momentum=0.6, nesterov=True)
            optimizers = [opt_adamw, opt_muon]
            desc = f"Muon(conv filters, lr={muon_lr}) + AdamW(others, lr={lr})"

    elif optimizer_name == "shampoo":
        opt = Shampoo(params_all, lr=lr, eps=1e-4, weight_decay=wd_effective,
                      max_dim=512)
        optimizers = [opt]
        desc = f"Simplified Shampoo(lr={lr}, max_dim=512)"

    else:
        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

    # 记录初始学习率，方便统一做线性衰减
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    return optimizers, desc

############################################
#                Training                  #
############################################

def main(arch_name, optimizer_name, model, train_loader, val_loader, eval_dataset,
         num_epochs, lr):
    device = next(model.parameters()).device

    optimizers, opt_desc = make_optimizers(model, optimizer_name, lr)

    # GPU 用 CUDA event 计时，CPU 用 wall clock
    use_cuda_timer = device.type == 'cuda'
    if use_cuda_timer:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    start_wall = None
    time_seconds = 0.0

    def start_timer():
        nonlocal start_wall
        if use_cuda_timer:
            starter.record()
        else:
            start_wall = time.perf_counter()

    def stop_timer():
        nonlocal time_seconds, start_wall
        if use_cuda_timer:
            ender.record()
            torch.cuda.synchronize()
            time_seconds += 1e-3 * starter.elapsed_time(ender)
        else:
            time_seconds += time.perf_counter() - start_wall

    model.reset()
    step = 0
    total_train_steps = num_epochs * len(train_loader)

    epoch_list = []
    train_acc_list = []
    val_acc_list = []
    time_list = []

    last_train_acc = None
    last_val_acc = None

    print(f"\n=== Arch: {arch_name}, Optimizer: {optimizer_name}, lr={lr} ===")
    print(f"{opt_desc}")

    run_name = f"{arch_name}/{optimizer_name}/lr={lr}"

    for epoch in range(num_epochs):
        start_timer()
        model.train()
        for obs, action in train_loader:
            obs = obs.to(device)
            action = action.to(device)

            logits = model(obs)          # (B,4)
            loss = F.cross_entropy(logits, action, reduction='mean')
            loss.backward()

            # 线性衰减 LR
            progress = step / max(1, total_train_steps)
            for opt in optimizers:
                for group in opt.param_groups:
                    init_lr = group["initial_lr"]
                    group["lr"] = init_lr * (1 - progress)

            for opt in optimizers:
                opt.step()

            model.zero_grad(set_to_none=True)
            step += 1
        stop_timer()

        # 最后一个 batch 的 train acc
        with torch.no_grad():
            pred_last = logits.argmax(1)
            last_train_acc = (pred_last == action).float().mean().item()

        last_val_acc = evaluate_acc(model, val_loader, device)

        epoch_list.append(epoch)
        train_acc_list.append(last_train_acc)
        val_acc_list.append(last_val_acc)
        time_list.append(time_seconds)

        variables = dict(
            run=run_name,
            epoch=epoch,
            train_acc=last_train_acc,
            val_acc=last_val_acc,
            maze_success=None,
            time_seconds=time_seconds,
        )
        print_training_details(variables, is_final_entry=False)
        run_name = ""  # 只在第一行打印 run 名字

    # greedy policy 迷宫成功率
    maze_success = evaluate_maze_success(model, eval_dataset, device)
    epoch = 'eval'
    variables = dict(
        run="",
        epoch=epoch,
        train_acc=None,
        val_acc=None,
        maze_success=maze_success,
        time_seconds=time_seconds,
    )
    print_training_details(variables, is_final_entry=True)

    return dict(
        arch_name=arch_name,
        optimizer_name=optimizer_name,
        lr=lr,
        train_acc=last_train_acc,
        val_acc=last_val_acc,
        maze_success=maze_success,
        time_seconds=time_seconds,
        optimizer_desc=opt_desc,
        epochs=epoch_list,
        train_accs=train_acc_list,
        val_accs=val_acc_list,
        time_seconds_per_epoch=time_list,
    )

############################################
#            Visualization helpers         #
############################################

def add_value_labels(ax, rects, fmt="{:.3f}"):
    """
    在柱状图每个柱子上标注数值。
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

def plot_same_opt_diff_lr(results, arch_name, opt_name, lrs, save_dir):
    """
    同一种架构 + 同一种优化器，不同学习率的最终 val_acc 柱状图。
    """
    vals = [results[(arch_name, opt_name, lr)]['val_acc'] for lr in lrs]
    fig, ax = plt.subplots()
    rects = ax.bar([str(lr) for lr in lrs], vals)
    ax.set_ylabel("Final val_acc")
    ax.set_xlabel("Learning rate")
    ax.set_title(f"Arch={arch_name}, Optimizer={opt_name}: val_acc vs lr")
    add_value_labels(ax, rects, fmt="{:.3f}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{arch_name}_val_acc_vs_lr_{opt_name}.png")
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

def plot_same_opt_epoch_curve(metrics, save_dir):
    """
    同一种架构 + 同一种优化器（某个 lr），随 epoch 的 train/val 精度曲线。
    """
    arch_name = metrics['arch_name']
    opt_name = metrics['optimizer_name']
    lr = metrics['lr']
    epochs = metrics['epochs']
    train_accs = metrics['train_accs']
    val_accs = metrics['val_accs']

    plt.figure()
    plt.plot(epochs, train_accs, marker='o', label='train_acc')
    plt.plot(epochs, val_accs, marker='s', label='val_acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Arch={arch_name}, Opt={opt_name}, lr={lr}: accuracy vs epoch")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"{arch_name}_epoch_curve_{opt_name}_lr{lr}.png")
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

def plot_diff_opt_same_lr(results, arch_name, optimizers, lr_ref, save_dir):
    """
    同一架构 + 固定 lr，不同优化器的最终 val_acc 柱状对比。
    """
    vals = [results[(arch_name, opt, lr_ref)]['val_acc'] for opt in optimizers]
    fig, ax = plt.subplots()
    rects = ax.bar(optimizers, vals)
    ax.set_ylabel("Final val_acc")
    ax.set_xlabel("Optimizer")
    ax.set_title(f"Arch={arch_name}, lr={lr_ref}: val_acc across optimizers")
    plt.xticks(rotation=45, ha='right')
    add_value_labels(ax, rects, fmt="{:.3f}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{arch_name}_val_acc_diff_opt_lr{lr_ref}.png")
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

def plot_best_maze_success(results, arch_name, optimizers, lrs, save_dir):
    """
    同一架构，不同优化器在各自最佳 lr 下的 maze_success 柱状图。
    """
    best_accs = []
    labels = []
    for opt in optimizers:
        best_s = -1.0
        best_lr = None
        for lr in lrs:
            s = results[(arch_name, opt, lr)]['maze_success']
            if s > best_s:
                best_s = s
                best_lr = lr
        best_accs.append(best_s)
        labels.append(f"{opt}\n(best_lr={best_lr})")

    fig, ax = plt.subplots()
    rects = ax.bar(labels, best_accs)
    ax.set_ylabel("Maze success (best lr)")
    ax.set_xlabel("Optimizer")
    ax.set_title(f"Arch={arch_name}: best maze_success per optimizer")
    plt.xticks(rotation=45, ha='right')
    add_value_labels(ax, rects, fmt="{:.3f}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{arch_name}_best_maze_success_per_optimizer.png")
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

def plot_time_diff_opt_same_lr(results, arch_name, optimizers, lr_ref, save_dir):
    """
    同一架构 + 固定 lr，不同优化器的训练时间柱状图。
    """
    times = [results[(arch_name, opt, lr_ref)]['time_seconds'] for opt in optimizers]
    fig, ax = plt.subplots()
    rects = ax.bar(optimizers, times)
    ax.set_ylabel("Total time (s)")
    ax.set_xlabel("Optimizer")
    ax.set_title(f"Arch={arch_name}, lr={lr_ref}: training time across optimizers")
    plt.xticks(rotation=45, ha='right')
    add_value_labels(ax, rects, fmt="{:.1f}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{arch_name}_time_diff_opt_lr{lr_ref}.png")
    plt.savefig(path)
    plt.close()
    print("Saved:", path)

############################################
#                 Entry                    #
############################################

if __name__ == "__main__":
    # 固定随机种子，保证不同架构/优化器/学习率用同一批迷宫
    torch.manual_seed(0)
    random.seed(0)

    num_epochs = 20             # 可以按需改小/改大
    LR_LIST = [0.001, 0.003, 0.01, 0.03]

    # 架构列表：CNN + Transformer + Mambo-style + Linear Attention
    ARCH_LIST = [
        "small", "deep", "res",
        "trans_small", "trans_deep",
        "mambo_small", "mambo_deep",
        "linattn_small",
    ]

    # 训练 / 验证 / 评估数据集
    train_set = MazeDataset(num_samples=500, H=15, W=15, wall_prob=0.2)
    val_set = MazeDataset(num_samples=100, H=15, W=15, wall_prob=0.2)
    eval_set = MazeDataset(num_samples=100, H=15, W=15, wall_prob=0.2)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 要对比的优化器列表
    optimizers_to_test = [
        "sgd", "sgd_no_mom",
        "adam", "adamw",
        "rmsprop", "lion",
        "muon", "muon_adamw",
        "shampoo",
    ]

    # 日志 & 图像保存目录
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)

    # 打印表头
    print_columns(logging_columns_list, is_head=True)

    # results[(arch, opt, lr)] = metrics
    results = {}

    for arch_name in ARCH_LIST:
        for opt_name in optimizers_to_test:
            for lr in LR_LIST:
                model = build_model(arch_name).to(device)
                metrics = main(arch_name, opt_name, model,
                               train_loader, val_loader, eval_set,
                               num_epochs=num_epochs, lr=lr)
                results[(arch_name, opt_name, lr)] = metrics

    # 汇总输出到控制台 + 写入 summary.txt
    print("\n=========== Summary across architectures, optimizers & lrs (Maze) ===========")
    header = ["arch", "optimizer", "lr", "train_acc", "val_acc", "maze_success", "time_seconds"]
    line = "|".join([h.center(12) for h in header])
    print("-" * len(line))
    print(line)
    print("-" * len(line))

    summary_lines = []
    summary_lines.append("=========== Summary across architectures, optimizers & lrs (Maze) ===========\n")
    summary_lines.append("-" * len(line) + "\n")
    summary_lines.append(line + "\n")
    summary_lines.append("-" * len(line) + "\n")

    for (arch_name, opt_name, lr), m in results.items():
        row = [
            arch_name.center(12),
            opt_name.center(12),
            f"{lr:.4g}".center(12),
            f"{m['train_acc']:.4f}".center(12),
            f"{m['val_acc']:.4f}".center(12),
            f"{m['maze_success']:.4f}".center(12),
            f"{m['time_seconds']:.2f}".center(12),
        ]
        row_str = "|".join(row)
        print(row_str)
        summary_lines.append(row_str + "\n")

    print("-" * len(line))
    summary_lines.append("-" * len(line) + "\n")

    summary_path = os.path.join(log_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.writelines(summary_lines)

    print("\nSummary saved to:")
    print(os.path.abspath(summary_path))

    # 逐个架构画图
    for arch_name in ARCH_LIST:
        # 1) 每个优化器内部，不同 lr 的 val_acc 对比
        for opt_name in optimizers_to_test:
            plot_same_opt_diff_lr(results, arch_name, opt_name, LR_LIST, log_dir)

        # 2) 每个优化器在该架构上“最佳 lr”下，随 epoch 的曲线
        for opt_name in optimizers_to_test:
            best_s = -1.0
            best_metrics = None
            for lr in LR_LIST:
                m = results[(arch_name, opt_name, lr)]
                if m['maze_success'] > best_s:
                    best_s = m['maze_success']
                    best_metrics = m
            if best_metrics is not None:
                plot_same_opt_epoch_curve(best_metrics, log_dir)

        # 3) 固定一个 lr，比不同优化器的 val_acc
        lr_ref = LR_LIST[2]  # 比如选 0.01
        plot_diff_opt_same_lr(results, arch_name, optimizers_to_test, lr_ref, log_dir)

        # 4) 不同优化器“最佳 lr”的 maze_success 对比
        plot_best_maze_success(results, arch_name, optimizers_to_test, LR_LIST, log_dir)

        # 5) 固定一个 lr，比不同优化器的训练时间
        plot_time_diff_opt_same_lr(results, arch_name, optimizers_to_test, lr_ref, log_dir)

    # 保存结果 + 代码
    log_path = os.path.join(log_dir, 'log_maze_compare_arch_optim_all_linear.pt')
    torch.save(dict(
        code=code,
        results=results,
        lr_list=LR_LIST,
        optimizers=optimizers_to_test,
        arch_list=ARCH_LIST,
        num_epochs=num_epochs),
        log_path
    )

    print("\nSaved results & figures in directory:")
    print(os.path.abspath(log_dir))

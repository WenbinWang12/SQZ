import torch
import math
from torch.optim import Optimizer

class Muon(Optimizer):
    """
    Standard Muon optimizer (hidden 2-D parameters) as per Jordan et al. (2024) write-up.
    Applies momentum + Newton-Schulz orthogonalization for matrix parameters, fallback to AdamW-like behavior for others.
    """
    def __init__(self, params, lr, momentum=0.95, weight_decay=0.0, ns_steps=5, eps=1e-7):
        # params: iterable of dicts or param groups
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz_ortho(G: torch.Tensor, steps: int, eps: float):
        # from blog post: implement approx orthogonalization of G (2‑D)
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.clone().detach()
        X = X.to(torch.float32)
        norm = X.norm() + eps
        X = X / norm
        # optionally transpose if more rows than cols
        transposed = False
        if X.size(0) > X.size(1):
            X = X.T
            transposed = True
        I = torch.eye(X.size(0), device=X.device, dtype=X.dtype)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if transposed:
            X = X.T
        return X.to(G.dtype)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # state initialization
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                # apply weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # choose update method
                if p.ndim == 2:
                    # apply Muon matrix update: orthogonalize momentum buffer
                    upd = self._newton_schulz_ortho(buf, ns_steps, eps)
                else:
                    # fallback: simple momentum SGD update (or you could integrate AdamW)
                    upd = buf / (buf.norm() + eps)

                # update param
                p.data.add_(upd, alpha=-lr)

        return loss


class ALSMuon:
    """
    Adaptive Layer-Selective Muon with Interval Scoring and Epsilon-Greedy.
    """
    def __init__(self, model, muon_param_filter, base_optimizer_factory,
                 B=8, muon_lr=1e-4, base_lr=1e-4, momentum=0.95, weight_decay=0.0, ns_steps=5, device=None, refresh_interval=10, epsilon=0.1):
        self.device = device
        self.model = model
        self.muon_params = []     # list of (name, param)
        self.base_params = []

        # Parameter Grouping
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if muon_param_filter(n, p):
                self.muon_params.append((n, p))
            else:
                self.base_params.append(p)

        self.total_muon_params = len(self.muon_params)
        self.B = min(B, max(1, self.total_muon_params))
        self.muon_lr = muon_lr

        # Interval & Strategy config
        self.refresh_interval = refresh_interval
        self.epsilon = epsilon
        self.step_counter = 0

        # Caching selection
        self.cached_selected_params = set()
        self.cached_selected_names = []

        # Set up Muon optimizer for muon_params
        muon_param_list = [p for _, p in self.muon_params]
        self.muon_optim = Muon([{'params': muon_param_list, 'lr': muon_lr,
                                 'momentum': momentum, 'weight_decay': weight_decay, 'ns_steps': ns_steps}])

        # Set up base optimizer (e.g. AdamW) for base_params
        self.base_optim = base_optimizer_factory(self.base_params, lr=base_lr)

    def step(self):
        # 1. 决定是否需要重新计算分数 (Re-score)
        needs_refresh = (self.step_counter % self.refresh_interval == 0)
        
        if needs_refresh:
            self._update_selection()
        
        # 2. 执行更新
        # 策略：对于未被选中的 Muon 参数，暂时隐藏其梯度，防止 Muon Optimizer 更新它们
        # 注意：这里我们假设 Muon Optimizer 会忽略 grad is None 的参数
        
        stashed_grads = {} # 用于恢复梯度（如果需要），或者单纯为了屏蔽
        
        # 屏蔽未选中的参数
        for name, p in self.muon_params:
            if p not in self.cached_selected_params:
                if p.grad is not None:
                    # 保存引用以防万一（虽然这里我们只做 step），
                    # 但更重要的是设为 None 让 Muon 跳过
                    stashed_grads[p] = p.grad
                    p.grad = None

        # Step Muon (只更新 Selected)
        self.muon_optim.step()

        # 恢复梯度 (为了后续的 zero_grad 或者统计)
        for p, g in stashed_grads.items():
            p.grad = g

        # Step Base (更新所有非 Muon 层)
        self.base_optim.step()

        self.step_counter += 1
        return self.cached_selected_names

    def _update_selection(self):
        """执行评分和选择逻辑"""
        scored = []
        # 只计算有梯度的参数
        for name, p in self.muon_params:
            if p.grad is None:
                continue
            # 优化点：不需要 detach().item()，保持 tensor 操作可能更快，但 item() 方便排序
            # 这里可以用更复杂的 metric，比如 Proposal 中的 spectral norm，这里先用 grad norm
            score = torch.norm(p.grad).item()
            scored.append((score, name, p))

        if not scored:
            self.cached_selected_params = set()
            self.cached_selected_names = []
            return

        # 按分数排序
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Epsilon-Greedy Logic
        n_random = int(self.B * self.epsilon)
        n_greedy = self.B - n_random
        
        # 确保索引不越界
        n_greedy = min(n_greedy, len(scored))
        
        # 1. 贪心部分
        greedy_candidates = scored[:n_greedy]
        
        # 2. 随机部分 (从剩余的中选)
        remaining_candidates = scored[n_greedy:]
        random_candidates = []
        if n_random > 0 and remaining_candidates:
            # 如果剩余不足 n_random，全选
            k = min(n_random, len(remaining_candidates))
            random_candidates = random.sample(remaining_candidates, k)
            
        selected = greedy_candidates + random_candidates
        
        # Update Cache
        self.cached_selected_names = [name for _, name, _ in selected]
        self.cached_selected_params = set([p for _, _, p in selected])

    def zero_grad(self):
        self.muon_optim.zero_grad()
        self.base_optim.zero_grad()
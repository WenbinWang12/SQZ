import time
import argparse
import yaml
from pathlib import Path
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

from optimizers import ALSMuon, FullMuon
from utils_logging import CSVLogger, SelectionRecorder

# ----------------------------- helper functions -----------------------------

def default_muon_filter(name, param):
    name = name.lower()
    if 'embed' in name or 'ln' in name or 'layernorm' in name:
        return False
    if name.endswith('.weight') and param.dim() >= 2 and param.numel() > 1024:
        return True
    return False


def build_dataloader(tokenizer, cfg, split='train'):
    # 使用本地缓存数据集（路径 ./data/wikitext-2-raw-v1）
    dataset_path = cfg.get('local_dataset_path', './data/wikitext-2-raw-v1')
    ds = load_dataset('text', data_files={'train': f'{dataset_path}/train.txt', 'validation': f'{dataset_path}/valid.txt'})[split]
    
    # Tokenize
    ds = ds.map(lambda x: tokenizer(x['text'], truncation=True, max_length=cfg['seq_len'], padding='max_length'), batched=True, remove_columns=ds.column_names)
    
    def collate_fn(batch):
        input_ids = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
        return input_ids
    
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=cfg.get('num_workers',0))
    return loader

# ----------------------------- main training -----------------------------

def train_main(config_path):
    cfg = yaml.safe_load(open(config_path))
    out_root = Path(cfg['output_dir']) / cfg['exp_name']
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用本地 tokenizer
    tokenizer_path = cfg.get('local_tokenizer_path', './local_gpt2_tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_cfg = GPT2Config(vocab_size=cfg.get('vocab_size', tokenizer.vocab_size), n_ctx=cfg['seq_len'], n_positions=cfg['seq_len'], n_layer=cfg.get('n_layer',12), n_embd=cfg.get('n_embd',768), n_head=cfg.get('n_head',12))
    model = GPT2LMHeadModel(model_cfg)
    model.to(device)

    train_loader = build_dataloader(tokenizer, cfg, split='train')

    optimizers_to_run = cfg['optimizers']

    # 全局时间追踪
    start_time = time.time()

    for opt_name in optimizers_to_run:
        print(f"Starting optimizer: {opt_name}")
        exp_dir = out_root / opt_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        csv_logger = CSVLogger(str(exp_dir), opt_name)
        selection_rec = SelectionRecorder()

        model_temp = GPT2LMHeadModel(model_cfg).to(device)

        if opt_name == 'AdamW':
            base_opt = AdamW(model_temp.parameters(), lr=cfg['lr'], betas=(0.9,0.95), eps=1e-8)
            muon_opt = None
        elif opt_name == 'FullMuon':
            muon_params = [p for n,p in model_temp.named_parameters() if default_muon_filter(n,p)]
            base_params = [p for n,p in model_temp.named_parameters() if not default_muon_filter(n,p)]
            muon_opt = FullMuon(muon_params, lr=cfg['muon_lr'])
            base_opt = AdamW(base_params, lr=cfg['lr'])
        elif opt_name == 'A-LS-Muon':
            base_factory = lambda params: AdamW(params, lr=cfg['lr'])
            muon_opt = ALSMuon(model_temp, muon_param_filter=default_muon_filter, base_optimizer_factory=base_factory, B=cfg['muon_B'], muon_lr=cfg['muon_lr'], device=device, ns_steps=cfg.get('ns_steps',1))
            base_opt = None
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        global_step = 0
        tokens_processed = 0

        print(f"Starting training loop for {opt_name}...")
        epoch_start_time = time.time()

        for epoch in range(cfg.get('epochs',1)):
            for batch in train_loader:
                step_start = time.time()

                input_ids = batch.to(device)
                labels = input_ids.clone()

                outputs = model_temp(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()

                # 计算全局 Gradient Norm (用于监控 Stability)
                # 这通常在 optimizer step 之前做。
                # 注意：这会增加一点计算开销，但对于分析稳定性至关重要
                total_norm = torch.nn.utils.clip_grad_norm_(model_temp.parameters(), max_norm=float('inf'))

                selected_names = []
                selected_ratio = 0.0

                if opt_name == 'AdamW':
                    base_opt.step()
                    base_opt.zero_grad()
                elif opt_name == 'FullMuon':
                    muon_opt.step()
                    base_opt.step()
                    muon_opt.zero_grad()
                    base_opt.zero_grad()
                    selected_ratio = 1.0 # 100% active
                elif opt_name == 'A-LS-Muon':
                    # ALSMuon 内部处理了 Base 和 Muon 的 step
                    selected_names = muon_opt.step()
                    muon_opt.zero_grad()
                    
                    # 计算 ratio
                    total_muon_candidates = max(1, len(muon_opt.muon_params))
                    selected_ratio = len(selected_names) / total_muon_candidates
                    
                    selection_rec.record(selected_names)

                global_step += 1
                tokens_processed += input_ids.numel()

                # [新增] 计算 Wall-clock
                current_wall_time = time.time() - start_time
                step_duration = time.time() - step_start

                if global_step % cfg.get('log_interval', 10) == 0:
                    print(f"[{opt_name}] Step {global_step}, Loss: {loss.item():.4f}, Tokens: {tokens_processed}, Selected_ratio: {selected_ratio}, GradNorm: {total_norm.item():.2f}, WallTime: {current_wall_time:.1f}s")

                csv_logger.log_step({
                    'step': global_step,
                    'epoch': epoch,
                    'loss': float(loss.item()),
                    'tokens': tokens_processed,
                    'grad_norm': float(total_norm.item()), 
                    'wall_seconds': float(current_wall_time), 
                    'step_time': float(step_duration), # 用于分析吞吐
                    'selected_ratio': selected_ratio,
                    'selected_names': str(selected_names)
                })

                if global_step >= cfg['total_steps']:
                    break
            if global_step >= cfg['total_steps']:
                break

        csv_logger.close()
        sel_csv = str(exp_dir / 'selection_heatmap.csv')
        selection_rec.selection_heatmap(sel_csv, param_order=[n for n,_ in model_temp.named_parameters()])
        print(f"Finished run {opt_name}. logs at {exp_dir}")

# ----------------------------- CLI -----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    train_main(args.config)

# ==================== 本地资源下载说明 ====================
# 1. 下载 GPT2 tokenizer 本地文件夹：
#    - 在可以访问 HuggingFace 的机器上执行：
#      from transformers import AutoTokenizer
#      tokenizer = AutoTokenizer.from_pretrained('gpt2')
#      tokenizer.save_pretrained('./local_gpt2_tokenizer')
#    - 将整个 local_gpt2_tokenizer 文件夹拷贝到目标机器
# 2. 下载 Wikitext-2-raw 数据集到本地：
#    - 官方下载链接：https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
#    - 解压到 ./data/wikitext-2-raw-v1/ 目录下
#    - 确保有 train.txt 和 valid.txt 文件
# 3. 在配置文件中指定路径：
#    local_tokenizer_path: './local_gpt2_tokenizer'
#    local_dataset_path: './data/wikitext-2-raw-v1'
# 4. 运行训练：
#    python train_gpt2_paper_local.py --config configs/gpt2_125m.yaml
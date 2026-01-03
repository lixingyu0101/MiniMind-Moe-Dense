import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import time

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    init_distributed_mode,
    setup_seed,
    init_model,
    get_lr,
    is_main_process,
    Logger
)

# =========================
# Train One Epoch
# =========================
def train_epoch(epoch, loader):
    model.train()

    num_experts = lm_config.n_routed_experts
    expert_counter = torch.zeros(num_experts, device=args.device)

    aux_loss_sum = 0.0
    aux_steps = 0

    loss_fct = nn.CrossEntropyLoss(reduction="none")

    for step, (X, Y, loss_mask) in enumerate(loader, start=1):
        X, Y, loss_mask = (
            X.to(args.device),
            Y.to(args.device),
            loss_mask.to(args.device)
        )

        lr = get_lr(
            epoch * len(loader) + step,
            args.epochs * len(loader),
            args.learning_rate
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = model(X)

            # ===== 专家负载统计 =====
            experts = out.expert_indices.reshape(-1)
            expert_counter.scatter_add_(
                0, experts,
                torch.ones_like(experts, dtype=torch.float)
            )

            # ===== LM Loss =====
            lm_loss = loss_fct(
                out.logits.view(-1, out.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            lm_loss = (lm_loss * loss_mask).sum() / loss_mask.sum()

            loss = lm_loss + out.aux_loss
            aux_loss_sum += out.aux_loss.detach().item()
            aux_steps += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ===== 日志 & 指标 =====
        if step % args.log_interval == 0:
            if dist.is_initialized():
                dist.all_reduce(expert_counter)

            if is_main_process():
                load = expert_counter.cpu().numpy()
                mean = load.mean()
                std = load.std()
                cv = std / (mean + 1e-8)
                max_mean = load.max() / (mean + 1e-8)
                aux_avg = aux_loss_sum / max(aux_steps, 1)

                Logger(
                    f"[{args.experiment_id}] "
                    f"CV={cv:.4f}, max/mean={max_mean:.3f}, "
                    f"aux_loss={aux_avg:.6f}"
                )

                os.makedirs(args.save_dir, exist_ok=True)
                with open(
                    f"{args.save_dir}/{args.experiment_id}_e{epoch}_s{step}.json",
                    "w"
                ) as f:
                    json.dump(
                        {
                            "experiment": args.experiment_id,
                            "epoch": epoch,
                            "step": step,
                            "expert_load": load.tolist(),
                            "cv": float(cv),
                            "max_mean_ratio": float(max_mean),
                            "aux_loss": float(aux_avg),
                            "aux_loss_alpha": args.aux_loss_alpha
                        },
                        f,
                        indent=2
                    )

            expert_counter.zero_()
            aux_loss_sum = 0.0
            aux_steps = 0


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--log_interval", type=int, default=100)

    # ⭐ 唯一控制负载均衡强度的参数
    parser.add_argument("--aux_loss_alpha", type=float, required=True)

    args = parser.parse_args()

    local_rank = init_distributed_mode()
    args.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    setup_seed(42)

    # ⭐⭐ 核心：aux_loss_alpha 必须传进 config
    lm_config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        use_moe=True,
        aux_loss_alpha=args.aux_loss_alpha
    )

    model, tokenizer = init_model(lm_config, from_weight="none", device=args.device)

    dataset = PretrainDataset(args.data_path, tokenizer)
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        train_epoch(epoch, loader)

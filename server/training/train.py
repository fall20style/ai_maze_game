"""LoRA-BERT 정책 학습.

    python -m training.train --data data/smoke2 --epochs 3 --out checkpoints/smoke

동작
----
1. ``train.npz`` / ``val.npz`` 를 시계열 Dataset 으로 감싼다.
2. bert-tiny 를 불러와 LoRA 를 주입하고, 투영층·헤드·LayerNorm 만 학습한다.
3. 검증 정확도가 가장 높은 체크포인트를 ``best.pt`` 로 저장한다.

주의: 정확도는 참고 지표일 뿐이다. 최종 판단은 ``training/evaluate.py`` 의
롤아웃 성공률로 한다 (클래스가 균형이어도 실제 주행은 실패할 수 있다).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.game.env import observation_dim
from app.ml.policy import DEFAULT_PRETRAINED, BertPolicy, PolicyConfig
from training.dataset import SequenceDataset


@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "checkpoints"
    epochs: int = 3
    batch_size: int = 64
    seq_len: int = 32
    lr: float = 1e-3
    lora_lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 0
    device: str = "cpu"
    pretrained: str | None = DEFAULT_PRETRAINED
    lora_r: int = 8
    lora_alpha: int = 16
    use_class_weights: bool = False
    log_every: int = 50
    init_checkpoint: str | None = None


def _load_obs_spec(data_dir: Path) -> tuple[int, int]:
    """meta.json 에서 (window_size, global_size) 를 읽는다."""
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = meta.get("config", {})
        return int(cfg.get("window_size", 3)), int(cfg.get("global_size", 5))
    return 3, 5


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_class_correct = torch.zeros(4, dtype=torch.long)
    per_class_total = torch.zeros(4, dtype=torch.long)

    with torch.inference_mode():
        for seq, mask, target in loader:
            seq, mask, target = seq.to(device), mask.to(device), target.to(device)
            pred = model(seq, mask).argmax(dim=-1)
            correct += int((pred == target).sum())
            total += target.numel()
            for cls in range(4):
                cls_mask = target == cls
                per_class_total[cls] += int(cls_mask.sum())
                per_class_correct[cls] += int((pred[cls_mask] == cls).sum())

    recall = (per_class_correct.float() / per_class_total.clamp(min=1).float()).tolist()
    return {
        "accuracy": correct / max(1, total),
        "samples": total,
        "per_class_recall": {n: round(r, 4) for n, r in zip(("UP", "DOWN", "LEFT", "RIGHT"), recall)},
    }


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_size, global_size = _load_obs_spec(data_dir)
    obs_dim = observation_dim(window_size, global_size)

    train_ds = SequenceDataset(data_dir / "train.npz", seq_len=cfg.seq_len)
    val_ds = SequenceDataset(data_dir / "val.npz", seq_len=cfg.seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    policy_cfg = PolicyConfig(
        obs_dim=obs_dim,
        seq_len=cfg.seq_len,
        pretrained=cfg.pretrained,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
    )
    model = BertPolicy(policy_cfg).to(cfg.device)

    if cfg.init_checkpoint:
        payload = BertPolicy.load_payload(cfg.init_checkpoint, map_location=cfg.device)
        model.load_state_dict(payload["state_dict"])
        print(f"[init] loaded weights from {cfg.init_checkpoint}")

    optimizer = torch.optim.AdamW(
        model.param_groups(base_lr=cfg.lr, lora_lr=cfg.lora_lr, weight_decay=cfg.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    weight = train_ds.class_weights().to(cfg.device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    print(f"[data] train={train_ds.describe()}")
    print(f"[data] val={val_ds.describe()}")
    print(f"[model] {model.stats.summary()}")

    history: list[dict] = []
    best_acc = -1.0
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        t0 = time.time()

        for step, (seq, mask, target) in enumerate(train_loader, start=1):
            seq, mask, target = seq.to(cfg.device), mask.to(cfg.device), target.to(cfg.device)

            logits = model(seq, mask)
            loss = criterion(logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running += float(loss.detach()) * target.size(0)
            seen += target.size(0)

            if cfg.log_every and step % cfg.log_every == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={running / max(1, seen):.4f}"
                )

        scheduler.step()

        train_loss = running / max(1, seen)
        val = evaluate_accuracy(model, val_loader, cfg.device)
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_accuracy": round(val["accuracy"], 5),
            "val_recall": val["per_class_recall"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(record)
        print(
            f"[epoch {epoch}] loss={train_loss:.4f} val_acc={val['accuracy']:.4f} "
            f"({record['seconds']}s)"
        )

        model.save(str(out_dir / "last.pt"), extra={"epoch": epoch, "history": history})
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            model.save(
                str(out_dir / "best.pt"),
                extra={"epoch": epoch, "val_accuracy": best_acc, "history": history},
            )

    total_time = round(time.time() - t_start, 1)
    summary = {
        "config": asdict(cfg),
        "policy_config": policy_cfg.to_dict(),
        "model_stats": model.stats.summary(),
        "best_val_accuracy": round(best_acc, 5),
        "total_seconds": total_time,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] best_val_acc={best_acc:.4f} total={total_time}s -> {out_dir}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA-BERT 미로 정책 학습")
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora-lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--init", type=str, default=None, help="이어서 학습할 체크포인트")
    p.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_const",
        const=None,
        help="사전학습 없이 랜덤 초기화 (비교 실험용)",
    )
    p.set_defaults(pretrained=DEFAULT_PRETRAINED)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = TrainConfig(
        data_dir=args.data,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        lora_lr=args.lora_lr,
        lora_r=args.lora_r,
        device=args.device,
        use_class_weights=args.class_weights,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=args.pretrained,
        log_every=args.log_every,
        init_checkpoint=args.init,
    )
    train(cfg)


if __name__ == "__main__":
    main()

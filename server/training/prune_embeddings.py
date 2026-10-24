"""쓰이지 않는 워드 임베딩을 잘라 체크포인트를 경량화한다.

배경
----
:class:`~app.ml.policy.BertPolicy` 는 항상 ``inputs_embeds`` 경로로 BERT 를
호출한다. HF ``BertEmbeddings.forward`` 는 ``inputs_embeds`` 가 주어지면
``word_embeddings`` 를 건너뛰므로, 30,522 × 128 어휘표는 **추론에 전혀 쓰이지
않는다**. 그런데도 ``state_dict`` 에 남아 체크포인트의 대부분을 차지한다.

이 스크립트는
1. 원본 체크포인트를 로드해 크기를 측정하고,
2. ``word_embeddings`` 를 1행 0 임베딩으로 교체한 뒤,
3. 같은 입력에 대한 로짓이 **완전히 동일**한지 확인하고,
4. 새 체크포인트와 before/after 표(JSON/Markdown)를 저장한다.

사용::

    python -m training.prune_embeddings --checkpoint checkpoints/wall/best.pt \
        --out checkpoints/wall_pruned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.ml.policy import BertPolicy

SERVER_ROOT = Path(__file__).resolve().parents[1]


def _state_bytes(state: dict) -> int:
    """state_dict 안 텐서들의 실제 바이트 수."""
    return sum(
        t.numel() * t.element_size() for t in state.values() if torch.is_tensor(t)
    )


def _mib(n: int) -> float:
    return n / (1024 * 1024)


def _print_safe(text: str) -> None:
    """cp949 콘솔에서도 깨지지 않게 출력한다 (파일은 UTF-8 그대로)."""
    import sys

    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _reference_inputs(
    cfg, samples: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """전체 길이 + 왼쪽 패딩 입력을 모두 만들어 본다 (러너와 동일한 형태)."""
    gen = torch.Generator().manual_seed(seed)
    full = torch.randn(samples, cfg.seq_len, cfg.obs_dim, generator=gen)
    full_mask = torch.ones(samples, cfg.seq_len, dtype=torch.long)

    # 러너는 시퀀스 앞쪽을 0으로 패딩하고 뒤쪽에 최근 관측을 채운다
    padded = torch.zeros_like(full)
    padded[:, -1:] = full[:, -1:]
    padded_mask = torch.zeros_like(full_mask)
    padded_mask[:, -1:] = 1

    half = torch.zeros_like(full)
    keep = max(1, cfg.seq_len // 2)
    half[:, -keep:] = full[:, -keep:]
    half_mask = torch.zeros_like(full_mask)
    half_mask[:, -keep:] = 1

    return [(full, full_mask), (padded, padded_mask), (half, half_mask)]


@torch.inference_mode()
def _collect_logits(model: BertPolicy, batches) -> list[torch.Tensor]:
    model.eval()
    return [model(x, mask).clone() for x, mask in batches]


def _size_row(label: str, before: float, after: float, unit: str = "") -> str:
    saved = before - after
    pct = 100 * saved / before if before else 0.0
    fmt = (lambda v: f"{v:,.0f}") if unit == "" else (lambda v: f"{v:,.2f}")
    return (
        f"| {label} | {fmt(before)}{unit} | {fmt(after)}{unit} "
        f"| {fmt(saved)}{unit} ({pct:.1f}%) |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="워드 임베딩 프루닝 + 용량 비교")
    parser.add_argument(
        "--checkpoint",
        default=str(SERVER_ROOT / "checkpoints" / "wall" / "best.pt"),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="출력 디렉터리 (기본: <checkpoint 부모>_pruned)",
    )
    parser.add_argument("--samples", type=int, default=8, help="동등성 검사 배치 크기")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        parser.error(f"체크포인트가 없습니다: {checkpoint}")

    out_dir = (
        Path(args.out)
        if args.out
        else checkpoint.parent.with_name(checkpoint.parent.name + "_pruned")
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = BertPolicy.load(str(checkpoint))

    before_params = sum(p.numel() for p in model.parameters())
    before_state_bytes = _state_bytes(model.state_dict())
    before_file_bytes = checkpoint.stat().st_size
    before_rows = int(model.encoder.embeddings.word_embeddings.weight.shape[0])

    batches = _reference_inputs(model.cfg, args.samples, args.seed)
    ref_logits = _collect_logits(model, batches)

    prune_info = model.prune_word_embeddings()
    new_logits = _collect_logits(model, batches)

    max_abs = max(
        float((a - b).abs().max().item()) for a, b in zip(ref_logits, new_logits)
    )
    exact = all(torch.equal(a, b) for a, b in zip(ref_logits, new_logits))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / checkpoint.name
    extra = dict(payload.get("extra") or {})
    extra["pruned_from"] = str(checkpoint)
    extra["prune"] = {
        "word_embeddings_rows_before": prune_info["before"]["rows"],
        "word_embeddings_rows_after": prune_info["after"]["rows"],
        "saved_params": prune_info["saved_params"],
        "logit_max_abs_diff": max_abs,
        "logit_exact": exact,
    }
    model.save(str(out_path), extra=extra)

    after_params = sum(p.numel() for p in model.parameters())
    after_state_bytes = _state_bytes(model.state_dict())
    after_file_bytes = out_path.stat().st_size

    report = {
        "checkpoint": str(checkpoint),
        "output": str(out_path),
        "word_embeddings": prune_info,
        "logit_equivalence": {
            "samples": args.samples,
            "batches": len(batches),
            "max_abs_diff": max_abs,
            "exact": exact,
        },
        "size": {
            "params": {
                "before": before_params,
                "after": after_params,
                "saved": before_params - after_params,
            },
            "state_bytes": {
                "before": before_state_bytes,
                "after": after_state_bytes,
                "saved": before_state_bytes - after_state_bytes,
            },
            "file_bytes": {
                "before": before_file_bytes,
                "after": after_file_bytes,
                "saved": before_file_bytes - after_file_bytes,
            },
        },
    }

    (out_dir / "prune_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# 워드 임베딩 프루닝 — {checkpoint.parent.name}",
        "",
        f"- 원본: `{checkpoint}`",
        f"- 결과: `{out_path}`",
        f"- 로짓 동등성: 최대 절대차 **{max_abs:.3e}** (완전 일치: {exact})",
        "",
        "| 항목 | Before | After | 절감 |",
        "| --- | ---: | ---: | ---: |",
        f"| word_embeddings 행 수 | {before_rows:,} | "
        f"{prune_info['after']['rows']:,} | "
        f"{before_rows - prune_info['after']['rows']:,} |",
        _size_row("전체 파라미터", before_params, after_params),
        _size_row("state_dict 텐서", _mib(before_state_bytes), _mib(after_state_bytes), " MiB"),
        _size_row("체크포인트 파일", _mib(before_file_bytes), _mib(after_file_bytes), " MiB"),
        "",
    ]
    (out_dir / "prune_report.md").write_text("\n".join(lines), encoding="utf-8")

    _print_safe("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

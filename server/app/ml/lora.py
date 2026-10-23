"""LoRA (Low-Rank Adaptation) 직접 구현.

``peft`` 를 쓰지 않는 이유
--------------------------
1. transformers 5.x 와의 호환성 리스크를 없앤다.
2. 어떤 모듈에 어떤 이름으로 붙었는지 명시적으로 확인할 수 있다 (디버깅 용이).
3. 의존성이 하나 줄어든다.

수식
----
    y = W x + (alpha / r) * B A x

- ``W`` 는 동결(freeze)된 원본 가중치
- ``A`` 는 ``(r, in_features)``, ``B`` 는 ``(out_features, r)``
- ``B`` 를 0으로 초기화하므로 학습 시작 시점의 출력은 원본과 **정확히** 같다
"""

from __future__ import annotations

import math

import torch
from torch import nn

#: BERT 에서 LoRA 를 붙일 리프 모듈 이름
#: query/key/value + (attention.output / intermediate / output) 의 dense
DEFAULT_TARGETS: tuple[str, ...] = ("query", "key", "value", "dense")


class LoRALinear(nn.Module):
    """동결된 ``nn.Linear`` + 학습 가능한 저랭크 잔차."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"r must be positive, got {r}")

        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B 는 0 → 학습 시작 시 원본과 동일한 출력
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + delta * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}"
        )


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> int:
    """``model`` 의 ``nn.Linear`` 을 제자리에서 :class:`LoRALinear` 로 교체한다.

    반환값은 교체된 모듈 수.
    """
    replaced = 0
    # named_modules 를 스냅샷으로 떠서 교체 중 순회 문제를 피한다
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue

        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced += 1

    return replaced


def mark_trainable(
    model: nn.Module,
    train_layernorm: bool = True,
    train_position_embeddings: bool = True,
) -> None:
    """LoRA 파라미터 + LayerNorm(+ 위치 임베딩)만 학습 가능하게 만든다.

    LayerNorm 을 여는 이유: 관측 투영층이 만드는 임베딩 분포는 BERT 사전학습 당시의
    토큰 임베딩과 크게 다르다. LayerNorm 의 scale/bias 만 풀어줘도 적응이 빨라진다.
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            continue

        trainable = False
        if train_layernorm and ("LayerNorm" in name or "layer_norm" in name):
            trainable = True
        if train_position_embeddings and "position_embeddings" in name:
            trainable = True

        param.requires_grad = trainable


def lora_parameter_count(model: nn.Module) -> int:
    return sum(
        p.numel() for n, p in model.named_parameters() if "lora_A" in n or "lora_B" in n
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_param_groups(
    model: nn.Module,
    base_lr: float,
    lora_lr: float,
    weight_decay: float = 0.01,
) -> list[dict]:
    """LoRA 파라미터와 나머지 학습 파라미터를 서로 다른 lr 로 묶는다."""
    lora, other = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (lora if ("lora_A" in name or "lora_B" in name) else other).append(param)

    groups: list[dict] = []
    if other:
        groups.append({"params": other, "lr": base_lr, "weight_decay": weight_decay})
    if lora:
        # LoRA 는 보통 weight decay 를 주지 않는다
        groups.append({"params": lora, "lr": lora_lr, "weight_decay": 0.0})
    return groups

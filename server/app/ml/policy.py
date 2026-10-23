"""BERT 기반 행동 정책.

구조
----
::

    obs_seq (B, T, obs_dim)
        │  Linear → LayerNorm → GELU → Dropout
        ▼
    inputs_embeds (B, T, hidden)
        │  BERT (LoRA 로 일부만 학습)
        ▼
    last_hidden (B, T, hidden)
        │  마지막 유효 토큰 선택
        ▼
    Linear(hidden, 4)  →  UP / DOWN / LEFT / RIGHT

**마지막 토큰만** 사용해 현재 스텝의 행동을 예측한다 (causal head).
시계열 전체를 보는 self-attention 이 곧 에이전트의 "기억" 역할을 한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from transformers import BertConfig, BertModel

from .lora import apply_lora, mark_trainable, split_param_groups

DEFAULT_PRETRAINED = "prajjwal1/bert-tiny"


@dataclass
class PolicyConfig:
    obs_dim: int = 54
    num_actions: int = 4
    seq_len: int = 32
    #: 관측 정의 — 체크포인트가 스스로를 설명하도록 함께 저장한다.
    #: (뷰어/서버가 같은 ``MazeEnv`` 설정으로 관측을 만들기 위해 필요)
    window_size: int = 3
    global_size: int = 5
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 512
    dropout: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    pretrained: str | None = DEFAULT_PRETRAINED
    train_layernorm: bool = True
    train_position_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PolicyStats:
    total: int = 0
    trainable: int = 0
    lora: int = 0
    lora_modules: int = 0

    def summary(self) -> str:
        pct = 100 * self.trainable / max(1, self.total)
        return (
            f"params total={self.total:,} trainable={self.trainable:,} ({pct:.2f}%) "
            f"lora={self.lora:,} over {self.lora_modules} modules"
        )


class BertPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        if cfg.pretrained:
            self.encoder = BertModel.from_pretrained(cfg.pretrained)
            if self.encoder.config.hidden_size != cfg.hidden_size:
                raise ValueError(
                    f"hidden size mismatch: checkpoint={self.encoder.config.hidden_size}, "
                    f"config={cfg.hidden_size}"
                )
        else:
            self.encoder = BertModel(
                BertConfig(
                    hidden_size=cfg.hidden_size,
                    num_hidden_layers=cfg.num_hidden_layers,
                    num_attention_heads=cfg.num_attention_heads,
                    intermediate_size=cfg.intermediate_size,
                    max_position_embeddings=max(512, cfg.seq_len),
                )
            )

        n_lora = apply_lora(
            self.encoder,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )
        mark_trainable(
            self.encoder,
            train_layernorm=cfg.train_layernorm,
            train_position_embeddings=cfg.train_position_embeddings,
        )

        self.head = nn.Linear(cfg.hidden_size, cfg.num_actions)

        self.stats = PolicyStats(
            total=sum(p.numel() for p in self.parameters()),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            lora=sum(
                p.numel()
                for n, p in self.named_parameters()
                if "lora_A" in n or "lora_B" in n
            ),
            lora_modules=n_lora,
        )

    # ------------------------------------------------------------------ 순전파
    def forward(
        self,
        obs_seq: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``obs_seq``: (B, T, obs_dim) → logits (B, num_actions)."""
        if obs_seq.dim() != 3:
            raise ValueError(f"obs_seq must be (B, T, D), got {tuple(obs_seq.shape)}")

        hidden = self.obs_proj(obs_seq)
        encoded = self.encoder(
            inputs_embeds=hidden, attention_mask=attention_mask
        ).last_hidden_state

        if attention_mask is None:
            last = encoded[:, -1]
        else:
            # 각 배치에서 마지막 유효 토큰을 고른다 (왼쪽 패딩 대응)
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            last = encoded[torch.arange(encoded.size(0), device=encoded.device), lengths]

        return self.head(last)

    # ------------------------------------------------------------------ 유틸
    def param_groups(self, base_lr: float, lora_lr: float, weight_decay: float = 0.01):
        return split_param_groups(self, base_lr, lora_lr, weight_decay)

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "config": self.cfg.to_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "BertPolicy":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(PolicyConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    @staticmethod
    def load_payload(path: str, map_location: str = "cpu") -> dict:
        return torch.load(path, map_location=map_location, weights_only=False)


def build_model(cfg: PolicyConfig) -> BertPolicy:
    return BertPolicy(cfg)

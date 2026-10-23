"""정책 실행기 — 관측 이력 버퍼를 관리한다.

학습 데이터는 "최근 K스텝 시퀀스"지만, 게임에서는 매 스텝 관측 하나만 들어온다.
이 클래스가 슬라이딩 윈도우를 유지해 **학습 때와 동일한 형태**의 입력을 만든다.

서버(W3)와 평가 스크립트가 같은 코드를 쓰므로 학습/서빙 불일치가 생기지 않는다.
"""

from __future__ import annotations

import numpy as np
import torch

from .policy import BertPolicy


class PolicyRunner:
    def __init__(
        self,
        model: BertPolicy,
        seq_len: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.seq_len = seq_len or model.cfg.seq_len
        self.obs_dim = model.cfg.obs_dim
        self.device = device
        self.buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self.buffer = []

    @property
    def history_length(self) -> int:
        return len(self.buffer)

    def _build_input(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(f"expected obs_dim={self.obs_dim}, got {obs.shape[0]}")

        self.buffer.append(obs)
        if len(self.buffer) > self.seq_len:
            del self.buffer[0 : len(self.buffer) - self.seq_len]

        seq = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.int64)
        n = len(self.buffer)
        seq[-n:] = np.stack(self.buffer)
        mask[-n:] = 1

        return (
            torch.from_numpy(seq).unsqueeze(0).to(self.device),
            torch.from_numpy(mask).unsqueeze(0).to(self.device),
        )

    @torch.inference_mode()
    def logits(self, obs: np.ndarray) -> torch.Tensor:
        seq, mask = self._build_input(obs)
        return self.model(seq, mask).squeeze(0)

    def probabilities(self, obs: np.ndarray) -> np.ndarray:
        probs = torch.softmax(self.logits(obs), dim=-1)
        return probs.cpu().numpy()

    def act(self, obs: np.ndarray, deterministic: bool = True, temperature: float = 1.0) -> int:
        logits = self.logits(obs)
        if deterministic:
            return int(logits.argmax().item())

        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

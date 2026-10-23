"""npz 데이터셋 → 시계열 Dataset.

각 샘플은 "현재 스텝까지의 최근 K개 관측" 시퀀스와 그 스텝의 정답 행동이다.

에피소드 경계
-------------
``step`` 컬럼이 0인 지점이 에피소드 시작이다. 시퀀스는 **에피소드를 넘어가지 않는다**
(이전 미로의 관측이 문맥으로 새어 들어가면 안 된다).

메모리
------
시퀀스를 미리 펼쳐 저장하지 않고 ``__getitem__`` 에서 슬라이스한다.
1M 샘플 × K=32 × 29차원을 미리 만들면 수 GB 가 되기 때문이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, npz_path: str | Path, seq_len: int = 32) -> None:
        data = np.load(npz_path)
        self.obs = np.ascontiguousarray(data["obs"], dtype=np.float32)
        self.action = np.ascontiguousarray(data["action"], dtype=np.int64)
        self.seq_len = seq_len
        self.n_samples = int(len(self.action))
        self.obs_dim = int(self.obs.shape[1]) if self.n_samples else 0

        if self.n_samples == 0:
            self.ep_start = np.zeros(0, dtype=np.int64)
            return

        steps = data["step"]
        starts = np.where(steps == 0)[0]
        if len(starts) == 0 or starts[0] != 0:
            starts = np.concatenate([[0], starts])

        # ep_start[i] = i 가 속한 에피소드의 시작 인덱스
        owner = np.searchsorted(starts, np.arange(self.n_samples), side="right") - 1
        self.ep_start = starts[owner].astype(np.int64)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lo = max(int(self.ep_start[index]), index - self.seq_len + 1)
        seq = self.obs[lo : index + 1]
        n = seq.shape[0]

        buf = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.int64)
        buf[self.seq_len - n :] = seq  # 왼쪽 패딩
        mask[self.seq_len - n :] = 1

        return (
            torch.from_numpy(buf),
            torch.from_numpy(mask),
            torch.tensor(self.action[index], dtype=torch.long),
        )

    # ------------------------------------------------------------------ 통계
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.action, minlength=4)

    def class_weights(self) -> torch.Tensor:
        """역빈도 가중치 (불균형이 남아 있을 때 대비)."""
        counts = self.class_counts().astype(np.float64)
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
        return torch.tensor(weights, dtype=torch.float32)

    def describe(self) -> dict:
        counts = self.class_counts().tolist()
        total = max(1, sum(counts))
        return {
            "samples": self.n_samples,
            "obs_dim": self.obs_dim,
            "episodes": int(len(np.unique(self.ep_start))),
            "action_share": [round(c / total, 3) for c in counts],
        }

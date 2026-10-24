"""워드 임베딩 프루닝 테스트.

핵심 불변식: ``inputs_embeds`` 경로만 쓰므로 임베딩을 1행으로 줄여도
로짓이 **비트 단위로 동일**해야 한다. 또한 저장/로드 왕복과 멱등성도 확인한다.
"""

from __future__ import annotations

import json

import pytest
import torch

from app.ml.policy import BertPolicy, PolicyConfig
from training.prune_embeddings import main as prune_main

VOCAB_ROWS = 30522  # BertConfig 기본 vocab_size


def _cfg(**overrides) -> PolicyConfig:
    base = dict(
        obs_dim=8,
        num_actions=4,
        seq_len=4,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        dropout=0.0,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        pretrained=None,
    )
    base.update(overrides)
    return PolicyConfig(**base)


def _inputs(model: BertPolicy, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    seq = torch.randn(3, model.cfg.seq_len, model.cfg.obs_dim, generator=gen)
    mask = torch.ones(3, model.cfg.seq_len, dtype=torch.long)
    mask[1, :2] = 0  # 왼쪽 패딩
    return seq, mask


def _logits(model: BertPolicy, seq, mask):
    model.eval()
    with torch.inference_mode():
        return model(seq, mask).clone()


def test_default_keeps_full_vocab():
    model = BertPolicy(_cfg())
    assert model.encoder.embeddings.word_embeddings.weight.shape == (VOCAB_ROWS, 16)
    assert model.cfg.prune_word_embeddings is False


def test_prune_keeps_logits_bit_exact():
    model = BertPolicy(_cfg())
    seq, mask = _inputs(model)
    before = _logits(model, seq, mask)

    info = model.prune_word_embeddings()
    after = _logits(model, seq, mask)

    assert info["before"]["rows"] == VOCAB_ROWS
    assert info["after"]["rows"] == 1
    assert torch.equal(before, after)


def test_prune_reduces_params_and_freezes_dummy():
    model = BertPolicy(_cfg())
    before = sum(p.numel() for p in model.parameters())

    info = model.prune_word_embeddings()
    after = sum(p.numel() for p in model.parameters())

    assert before - after == (VOCAB_ROWS - 1) * 16
    assert info["saved_params"] == (VOCAB_ROWS - 1) * 16
    assert info["saved_bytes"] > 0
    dummy = model.encoder.embeddings.word_embeddings
    assert dummy.weight.shape == (1, 16)
    assert not dummy.weight.requires_grad
    assert torch.count_nonzero(dummy.weight) == 0
    assert model.stats.total == after


def test_prune_is_idempotent():
    model = BertPolicy(_cfg())
    first = model.prune_word_embeddings()
    params = sum(p.numel() for p in model.parameters())

    second = model.prune_word_embeddings()

    assert first["already_pruned"] is False
    assert second["already_pruned"] is True
    assert second["saved_params"] == 0
    assert sum(p.numel() for p in model.parameters()) == params


def test_init_flag_prunes_before_load():
    model = BertPolicy(_cfg(prune_word_embeddings=True))
    assert model.encoder.embeddings.word_embeddings.weight.shape == (1, 16)


def test_save_load_roundtrip(tmp_path):
    model = BertPolicy(_cfg())
    seq, mask = _inputs(model)
    expected = _logits(model, seq, mask)
    model.prune_word_embeddings()

    path = tmp_path / "pruned.pt"
    model.save(str(path))

    loaded = BertPolicy.load(str(path))
    assert loaded.cfg.prune_word_embeddings is True
    assert loaded.encoder.embeddings.word_embeddings.weight.shape == (1, 16)
    assert torch.equal(_logits(loaded, seq, mask), expected)


def test_cli_writes_report(tmp_path, capsys):
    model = BertPolicy(_cfg())
    seq, mask = _inputs(model)
    expected = _logits(model, seq, mask)
    ckpt = tmp_path / "best.pt"
    model.save(str(ckpt))

    out_dir = tmp_path / "pruned"
    assert prune_main(
        ["--checkpoint", str(ckpt), "--out", str(out_dir), "--samples", "2"]
    ) == 0

    assert (out_dir / "best.pt").exists()
    report = json.loads((out_dir / "prune_report.json").read_text(encoding="utf-8"))
    assert report["logit_equivalence"]["exact"] is True
    assert report["logit_equivalence"]["max_abs_diff"] == pytest.approx(0.0)
    assert report["size"]["params"]["after"] < report["size"]["params"]["before"]
    assert report["size"]["file_bytes"]["after"] < report["size"]["file_bytes"]["before"]
    assert "| 항목 | Before | After | 절감 |" in (out_dir / "prune_report.md").read_text(
        encoding="utf-8"
    )

    loaded = BertPolicy.load(str(out_dir / "best.pt"))
    assert torch.equal(_logits(loaded, seq, mask), expected)

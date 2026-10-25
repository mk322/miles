"""Tests for ``run_qwen3_8b_ota_sft.py``: the train argv it submits (no GPU, ray or checkpoint needed).

Run inside the Miles image: python -m pytest examples/ota_sft/tests/test_launcher.py -q
"""

import json
import os
import shlex
from pathlib import Path

import pytest
import typer.testing

from examples.ota_sft import run_qwen3_8b_ota_sft as launcher
from tests.fast.utils.command_recorder import record_commands

TEMPLATE = Path(launcher.__file__).resolve().parent / "qwen3_chat_template.jinja"
QWEN3_8B = Path(os.environ.get("QWEN3_8B", "/root/models/Qwen3-8B"))


def _train_argv(monkeypatch, *cli: str) -> list[str]:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    commands = record_commands(monkeypatch)
    result = typer.testing.CliRunner().invoke(launcher.app, ["train", *cli])
    assert result.exit_code == 0, result.output
    submit = [c for c in commands if "train_async.py" in c]
    assert len(submit) == 1, commands
    return shlex.split(submit[0].split("train_async.py", 1)[1])


def _value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, (flag, argv)
    return argv[argv.index(flag) + 1]


def test_full_tree_run_uses_tp2_and_the_committed_template(monkeypatch):
    argv = _train_argv(monkeypatch, "--finetune-type", "full", "--sft-tree-attention",
                       "--data-file", "ota_sft_10k_aligned.jsonl", "--save-interval", "104")
    assert _value(argv, "--tensor-model-parallel-size") == "2"
    assert "--sequence-parallel" in argv
    assert _value(argv, "--lr") == "4e-05"
    assert _value(argv, "--clip-grad") == "1.0"
    assert _value(argv, "--weight-decay") == "0.04"
    assert _value(argv, "--global-batch-size") == "96"
    assert _value(argv, "--num-epoch") == "7"
    assert _value(argv, "--save-interval") == "104"
    assert _value(argv, "--prompt-data").endswith("/ota_sft_10k_aligned.jsonl")
    assert "--sft-tree-attention" in argv
    assert Path(_value(argv, "--sft-tree-chat-template")).resolve() == TEMPLATE


def test_explicit_template_and_tp_override(monkeypatch):
    argv = _train_argv(monkeypatch, "--finetune-type", "full", "--sft-tree-attention",
                       "--chat-template", "/x/t.jinja", "--tensor-model-parallel-size", "4")
    assert _value(argv, "--sft-tree-chat-template") == "/x/t.jinja"
    assert _value(argv, "--tensor-model-parallel-size") == "4"


def test_no_tree_flags_without_tree_attention(monkeypatch):
    argv = _train_argv(monkeypatch, "--finetune-type", "lora")
    assert "--sft-tree-attention" not in argv and "--sft-tree-chat-template" not in argv
    assert _value(argv, "--lora-rank") == "64"


@pytest.mark.skipif(not (QWEN3_8B / "tokenizer_config.json").exists(), reason="Qwen3-8B not available")
def test_committed_template_is_qwen3_8b_own_template():
    # the eval server renders with the checkpoint's template, which the HF export copies from Qwen3-8B
    assert TEMPLATE.read_text() == json.loads((QWEN3_8B / "tokenizer_config.json").read_text())["chat_template"]

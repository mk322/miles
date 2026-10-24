"""Activation memory and step time of Qwen3-8B-shaped layers under tree attention, on one GPU.

Builds a random-weight Megatron GPTModel with Qwen3-8B widths (hidden 4096, 32/8 heads, ffn 12288,
vocab 151936) and ``--layers`` decoder layers at TP1, installs tree attention, and runs fwd+bwd of
one micro-batch of a realistic tree of ``--tokens`` tokens with full recompute (uniform, 1 layer
per chunk, i.e. every layer recomputed) or none. Reports peak memory above the static weights+grads
and the step time, so per-layer and per-token costs can be fitted and extrapolated to 36 layers and
TP > 1 (see NOTES.md).

    CUDA_VISIBLE_DEVICES=<gpu> torchrun --nproc_per_node=1 tests/manual/sft_tree/probe_tree_memory.py --layers 2 --tokens 32768
"""

import argparse
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.backends.megatron_utils.tree_attention import get_tree_packed_seq_params, install_tree_attention
from miles.backends.training_utils.data.sft_tree_layout import build_tree_batch_layout
from miles.backends.training_utils.loss.hub.math_utils import calculate_log_probs_and_entropy


def _tree(tokens: int):
    """A tree shaped like an OTA trace: trunk 60% of the tokens, ~12 branches of equal length."""
    n_branch = 11
    blen = int(tokens * 0.4 / (n_branch + 1))
    seg0 = tokens - n_branch * blen
    trunk = seg0 - blen
    prefixes = [int(trunk * (i + 1) / (n_branch + 1)) for i in range(n_branch)]
    return seg0, [(blen, p) for p in prefixes]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--recompute", choices=["full", "none"], default="full")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--plain", action="store_true", help="one causal sequence instead of a tree (no branches)")
    args = p.parse_args()

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    parallel_state.initialize_model_parallel(1)
    model_parallel_cuda_manual_seed(0)
    from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
    from miles.utils.ft_utils.process_group_utils import GroupInfo

    one = GroupInfo(rank=0, size=1, group=parallel_state.get_data_parallel_group())
    set_parallel_state(ParallelState(intra_dp=one, intra_dp_cp=one, cp=one, tp=one, pp=one, ep=one, etp=one, indep_dp=one))

    recompute = {} if args.recompute == "none" else dict(
        recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
    )
    config = TransformerConfig(
        num_layers=args.layers, hidden_size=4096, ffn_hidden_size=12288, num_attention_heads=32,
        num_query_groups=8, kv_channels=128, normalization="RMSNorm", layernorm_epsilon=1e-6,
        gated_linear_unit=True, activation_func=F.silu, add_bias_linear=False, qk_layernorm=True,
        hidden_dropout=0.0, attention_dropout=0.0, bf16=True, params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash, apply_rope_fusion=True, variable_seq_lengths=True,
        calculate_per_token_loss=True, **recompute,
    )
    model = GPTModel(
        config=config, transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
        vocab_size=151936, max_sequence_length=131072, position_embedding_type="rope", rotary_base=1000000,
        share_embeddings_and_output_weights=False, parallel_output=True,
    ).cuda()
    install_tree_attention(model)
    # stand-ins for Megatron's fp32 main_grad buffers (static memory; the probe measures what is above it)
    main_grads = [torch.zeros_like(prm, dtype=torch.float32) for prm in model.parameters()]  # noqa: F841

    seg0, branches = _tree(args.tokens) if not args.plain else (args.tokens, [])
    total = seg0 + sum(b for b, _ in branches)
    layout = build_tree_batch_layout([total], [branches], pad=0)
    psp = get_tree_packed_seq_params(layout)
    tokens = torch.randint(0, 151936, (1, total), device="cuda")

    def step():
        logits = model(input_ids=tokens, position_ids=None, attention_mask=None, labels=None, packed_seq_params=psp)
        lp, _ = calculate_log_probs_and_entropy(
            logits[0, :-1], tokens[0, 1:], parallel_state.get_tensor_model_parallel_group(),
            with_entropy=False, chunk_size=args.chunk,
        )
        lp.sum().backward()

    step()
    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 3
    out = {
        "layers": args.layers, "tokens": total, "recompute": args.recompute, "tree": not args.plain,
        "static_gib": round(static / 2**30, 2),
        "peak_above_static_gib": round((torch.cuda.max_memory_allocated() - static) / 2**30, 2),
        "step_s": round(dt, 3), "tokens_per_s": round(total / dt),
    }
    print("RESULT " + json.dumps(out), flush=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    main()

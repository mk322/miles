"""Model-level golden test of ``--sft-tree-attention`` through the real Megatron + Miles path.

A tiny Qwen3-shaped GPTModel (TE layers, GQA, qk-layernorm, RoPE base 1e6, full recompute) is run
twice from identical weights over the same random multi-turn traces:

* expanded: every supervised turn as its own standalone sample, through the existing path
  (TE flash attention, Megatron RoPE);
* tree: the traces packed as trees (``miles.rollout.sft_tree``) through tree attention, packed
  several per micro-batch in a different micro-batch grouping, with a fully masked placeholder.

Both go through Miles' ``get_batch`` / ``get_packed_seq_params`` / ``loss_function`` (sft_loss,
per-token loss) and are reduced the way Megatron's per-token finalize does (sum of losses and
grads over micro-batches, divided by the step's total ``num_tokens``). Checked: per-token
log-probs of every supervised token, the step's token count, the step loss, and every parameter's
gradient, each against the noise floor of the expanded path itself (the same samples regrouped).
One-turn trees must be bitwise identical to the existing path.

    python tests/fast-gpu/test_sft_tree_attention_model.py   # runs TP1 then TP2 (+ sequence parallel)
    NVIDIA_TF32_OVERRIDE=0 torchrun --nproc_per_node=1 tests/fast-gpu/test_sft_tree_attention_model.py  # one TP size
"""

import json
import os
import random
import subprocess
import sys
from argparse import Namespace
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
from tests.ci.ci_register import register_cuda_ci

from miles.backends.megatron_utils import tree_attention
from miles.backends.megatron_utils.parallel import get_packed_seq_params
from miles.backends.megatron_utils.tree_attention import install_tree_attention
from miles.backends.training_utils.data.rollout import DataIterator, get_batch
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.rollout.sft_tree import TurnSample, build_trees, expand_tree
from miles.utils.ft_utils.process_group_utils import GroupInfo

register_cuda_ci(
    est_time=600, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=2
)

VOCAB = 512
LAYERS = 3


def _group(group) -> GroupInfo:
    return GroupInfo(rank=dist.get_rank(group), size=dist.get_world_size(group), group=group)


def _set_miles_parallel_state() -> None:
    tp = _group(parallel_state.get_tensor_model_parallel_group())
    one = GroupInfo(rank=0, size=1, group=parallel_state.get_data_parallel_group())
    set_parallel_state(
        ParallelState(
            intra_dp=one,
            intra_dp_cp=one,
            cp=GroupInfo(rank=0, size=1, group=parallel_state.get_context_parallel_group()),
            tp=tp,
            pp=GroupInfo(rank=0, size=1, group=parallel_state.get_pipeline_model_parallel_group()),
            ep=one,
            etp=one,
            indep_dp=one,
        )
    )


def _config(tp: int, *, rope_fusion: bool, fp32: bool = False) -> TransformerConfig:
    return TransformerConfig(
        num_layers=LAYERS,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=64,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=not fp32,
        params_dtype=torch.float32 if fp32 else torch.bfloat16,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        attention_backend=AttnBackend.flash,
        apply_rope_fusion=rope_fusion,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=1,
    )


def _model(config: TransformerConfig) -> GPTModel:
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
        vocab_size=VOCAB,
        max_sequence_length=8192,
        position_embedding_type="rope",
        rotary_base=1000000,
        share_embeddings_and_output_weights=False,
        parallel_output=True,
    ).cuda()
    return model


def _args(tree: bool) -> Namespace:
    return Namespace(
        loss_type="sft_loss",
        calculate_per_token_loss=True,
        qkv_format="thd",
        allgather_cp=False,
        log_probs_chunk_size=128,
        log_probs_backend="torch",
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        recompute_loss_function=False,
        use_dynamic_global_batch_size=False,
        global_batch_size=1,
        sft_tree_attention=tree,
        vocab_size=VOCAB,
        debug_unified_grad_fused_logprob=False,
        use_rollout_entropy=False,
        data_pad_size_multiplier=128,
        multi_lora_n_adapters=None,
        lora_adapter_paths=None,
        multi_lora=False,
    )


# ------------------------------------------------------------------ synthetic traces


def _random_trace(rng: random.Random, n_turns: int) -> list[TurnSample]:
    """Turn samples whose prompts nest like a real trace's: prompt_{k+1} = prompt_k + stripped answer + observation."""
    prompt = [rng.randrange(VOCAB) for _ in range(rng.randint(20, 300))]
    turns = []
    for k in range(n_turns):
        target = [rng.randrange(VOCAB) for _ in range(rng.randint(2, 200))]
        turns.append(TurnSample(message_index=2 * k + 1, prompt_ids=tuple(prompt), target_ids=tuple(target)))
        prompt = prompt + target[-rng.randint(1, len(target)) :] + [rng.randrange(VOCAB) for _ in range(rng.randint(5, 150))]
    return turns


def _samples(rng: random.Random, n_traces: int, max_tree_tokens: int):
    trees, expanded = [], []
    for _ in range(n_traces):
        turns = _random_trace(rng, rng.randint(1, 7))
        t, _ = build_trees(turns, max_context_len=max_tree_tokens, max_tree_tokens=max_tree_tokens)
        trees += t
    for tree in trees:
        for tokens, mask in expand_tree(tree):
            first = mask.index(1)
            expanded.append({"tokens": tokens, "response_length": len(tokens) - first, "loss_mask": mask[first:]})
    tree_samples = [
        {
            "tokens": list(t.tokens),
            "response_length": t.response_length,
            "loss_mask": list(t.loss_mask),
            "tree_branches": [list(b) for b in t.branches],
        }
        for t in trees
    ]
    return tree_samples, expanded


def _placeholder() -> dict:
    return {"tokens": [1, 2], "response_length": 1, "loss_mask": [0], "tree_branches": []}


# ------------------------------------------------------------------ one step


def _rollout_data(samples: list[dict], micro_batches: list[list[int]]):
    device = torch.cuda.current_device()
    data = {
        "tokens": [torch.tensor(s["tokens"], dtype=torch.long, device=device) for s in samples],
        "loss_masks": [torch.tensor(s["loss_mask"], dtype=torch.int, device=device) for s in samples],
        "response_lengths": [s["response_length"] for s in samples],
        "total_lengths": [len(s["tokens"]) for s in samples],
    }
    if any("tree_branches" in s for s in samples):
        data["tree_branches"] = [s.get("tree_branches") for s in samples]
    return DataIterator(data, micro_batch_indices=micro_batches)


def _step(model, args, samples, micro_batches):
    """Forward/backward every micro-batch like Miles' train step; returns per-sample log-probs, loss sum, tokens."""
    model.zero_grad(set_to_none=True)
    it = _rollout_data(samples, micro_batches)
    keys = ["tokens", "total_lengths", "response_lengths", "loss_masks", "tree_branches"]
    log_probs, loss_sum, num_tokens, grads = {}, 0.0, 0, {}
    for mb in micro_batches:
        batch = get_batch(it, keys, args.data_pad_size_multiplier, "thd", sft_tree_attention=args.sft_tree_attention)
        logits = model(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=get_packed_seq_params(batch, args),
            loss_mask=batch["input_loss_masks"],
        )
        with torch.no_grad():
            lp = get_log_probs_and_entropy(
                logits.detach(),
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
            )["log_probs"]
        for i, x in zip(mb, lp, strict=True):
            log_probs[i] = x.float()
        loss, n_tok, _ = loss_function(args, batch, len(micro_batches), logits, apply_megatron_loss_scaling=True)
        loss.backward()
        loss_sum += loss.item()
        num_tokens += int(n_tok.item())
        # accumulate in fp32 across micro-batches, as Megatron's main_grad buffers do
        for n, p in model.named_parameters():
            if p.grad is not None:
                grads[n] = grads.get(n, 0) + p.grad.float()
                p.grad = None
    # Megatron's per-token finalize: grads (summed over micro-batches) / the step's total num_tokens
    grads = {n: g / num_tokens for n, g in grads.items()}
    return log_probs, loss_sum / num_tokens, num_tokens, grads


def _group_micro_batches(samples, budget, rng):
    order = list(range(len(samples)))
    rng.shuffle(order)
    mbs, cur, size = [], [], 0
    for i in order:
        n = len(samples[i]["tokens"])
        if cur and size + n > budget:
            mbs.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    mbs.append(cur)
    return mbs


def _target_log_probs(samples, log_probs):
    """Per-sample log-probs of the supervised tokens only, concatenated in a canonical order."""
    out = []
    for i, s in enumerate(samples):
        mask = torch.tensor(s["loss_mask"], device=log_probs[i].device).bool()
        out.append(log_probs[i][mask])
    return out


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def run(tp: int):
    rank = dist.get_rank()
    rng = random.Random(0)
    tree_samples, expanded = _samples(rng, n_traces=12, max_tree_tokens=1500)
    report = {"tp": tp, "trees": len(tree_samples), "expanded_samples": len(expanded)}

    # --- one-turn trees are the existing path, bitwise
    single = [s for s in tree_samples if not s["tree_branches"]]
    assert single, "need some one-turn trees"
    for rope_fusion in (False, True):
        base, tree_model = _model(_config(tp, rope_fusion=rope_fusion)), _model(_config(tp, rope_fusion=rope_fusion))
        tree_model.load_state_dict(base.state_dict())
        install_tree_attention(tree_model)
        mbs = [list(range(len(single)))]
        lp_a, loss_a, n_a, g_a = _step(base, _args(False), single, mbs)
        lp_b, loss_b, n_b, g_b = _step(tree_model, _args(True), single, mbs)
        diff = max((lp_a[i] - lp_b[i]).abs().max().item() for i in lp_a)
        gdiff = max((g_a[n] - g_b[n]).abs().max().item() for n in g_a)
        report[f"single_turn_rope_fusion={rope_fusion}"] = {"max_abs_logprob_diff": diff, "max_abs_grad_diff": gdiff}
        assert diff == 0 and gdiff == 0 and loss_a == loss_b and n_a == n_b, report

    # --- bf16, production kernels: trees through tree attention vs expanded samples through the existing path
    base, tree_model = _model(_config(tp, rope_fusion=True)), _model(_config(tp, rope_fusion=True))
    tree_model.load_state_dict(base.state_dict())
    install_tree_attention(tree_model)
    bf16 = _compare(base, _args(False), tree_model, tree_samples, expanded)
    report["bf16"] = bf16
    assert abs(bf16["loss_tree"] - bf16["loss_expanded"]) <= 1e-4 * abs(bf16["loss_expanded"]), report
    assert bf16["max_abs_logprob_diff"] <= 3 * bf16["max_abs_logprob_diff_noise_floor"] + 1e-2, report
    assert bf16["max_grad_rel"] <= 3 * bf16["max_grad_rel_noise_floor"] + 1e-2, report

    # --- fp32 with exact (fp64 dense masked) attention in both: isolates the tree construction itself
    # (positions, masks, labels, normalization) from kernel rounding. The expanded samples run through
    # TreeCoreAttention too, which on plain causal sequences is the existing path bit for bit (above).
    torch.backends.cuda.matmul.allow_tf32 = False
    original = tree_attention.tree_flash_attention
    tree_attention.tree_flash_attention = _dense_tree_attention
    try:
        base, tree_model = _model(_config(tp, rope_fusion=True, fp32=True)), _model(_config(tp, rope_fusion=True, fp32=True))
        tree_model.load_state_dict(base.state_dict())
        install_tree_attention(base)
        install_tree_attention(tree_model)
        fp32 = _compare(base, _args(True), tree_model, tree_samples, expanded)
    finally:
        tree_attention.tree_flash_attention = original
    report["fp32_exact_attention"] = fp32
    if rank == 0:
        print(json.dumps(report, indent=1), flush=True)
    assert abs(fp32["loss_tree"] - fp32["loss_expanded"]) <= 1e-6 * abs(fp32["loss_expanded"]), report
    assert fp32["max_abs_logprob_diff"] <= 1e-5, report
    assert fp32["max_grad_rel"] <= 3 * fp32["max_grad_rel_noise_floor"] + 1e-5, report
    assert fp32["total_grad_rel"] <= 3 * fp32["total_grad_rel_noise_floor"] + 1e-6, report
    return report


def _dense_tree_attention(q, k, v, layout, *, softmax_scale, deterministic=False):
    """Tree attention as fp64 dense masked softmax (autograd), for the exact-arithmetic comparison."""
    cu = layout.cu_seqlens.tolist()
    T = q.shape[0]
    mask = torch.zeros(T, T, dtype=torch.bool, device=q.device)
    for a, b in zip(cu[:-1], cu[1:], strict=True):
        mask[a:b, a:b] = torch.ones(b - a, b - a, dtype=torch.bool, device=q.device).tril()
    for qs, qe, ks, ke in layout.cross:
        mask[qs:qe, ks:ke] = True
    group = q.shape[1] // k.shape[1]
    q64 = q.double().transpose(0, 1)  # [h, T, d]
    k64 = k.double().repeat_interleave(group, dim=1).transpose(0, 1)
    v64 = v.double().repeat_interleave(group, dim=1).transpose(0, 1)
    s = (q64 @ k64.transpose(1, 2)) * softmax_scale
    p = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
    return (p @ v64).transpose(0, 1).to(q.dtype).contiguous()


def _compare(base, base_args, tree_model, tree_samples, expanded) -> dict:
    exp_mbs = _group_micro_batches(expanded, 4096, random.Random(1))
    lp_e, loss_e, n_e, g_e = _step(base, base_args, expanded, exp_mbs)
    # noise floor: the same expanded samples, grouped into other micro-batches
    lp_e2, loss_e2, n_e2, g_e2 = _step(base, base_args, expanded, _group_micro_batches(expanded, 3000, random.Random(2)))

    # trees in yet another grouping, plus a fully masked placeholder (a trace with no trainable turn)
    trees = tree_samples + [_placeholder()]
    lp_t, loss_t, n_t, g_t = _step(tree_model, _args(True), trees, _group_micro_batches(trees, 4096, random.Random(3)))

    # supervised log-probs, mapped tree -> expanded order (expand_tree order: inline turn, then branches)
    exp_targets = _target_log_probs(expanded, lp_e)
    exp2_targets = _target_log_probs(expanded, lp_e2)
    flat_tree, cursor = [], 0
    for s, lp in zip(trees, _target_log_probs(trees, lp_t), strict=True):
        n_turns = len(s["tree_branches"]) + 1 if s["loss_mask"] != [0] else 0
        sizes = [len(x) for x in exp_targets[cursor : cursor + n_turns]]
        flat_tree += list(lp.split(sizes)) if n_turns else []
        cursor += n_turns
    assert cursor == len(expanded)
    # the step's per-token normalizer is the expanded samples' token count; the placeholder adds nothing
    assert n_t == n_e == n_e2 == sum(sum(s["loss_mask"]) for s in expanded), (n_t, n_e, n_e2)
    grad_rel = {n: _rel(g_t[n], g_e[n]) for n in g_e}
    grad_rel_noise = {n: _rel(g_e2[n], g_e[n]) for n in g_e}
    assert set(g_t) == set(g_e)
    return {
        "num_tokens": n_t,
        "loss_tree": loss_t,
        "loss_expanded": loss_e,
        "loss_expanded_regrouped": loss_e2,
        "max_abs_logprob_diff": max((a - b).abs().max().item() for a, b in zip(flat_tree, exp_targets, strict=True)),
        "max_abs_logprob_diff_noise_floor": max(
            (a - b).abs().max().item() for a, b in zip(exp2_targets, exp_targets, strict=True)
        ),
        "max_grad_rel": max(grad_rel.values()),
        "max_grad_rel_noise_floor": max(grad_rel_noise.values()),
        "worst_grad": max(grad_rel, key=grad_rel.get),
        "total_grad_rel": _rel(torch.cat([g_t[n].flatten() for n in g_e]), torch.cat([g_e[n].flatten() for n in g_e])),
        "total_grad_rel_noise_floor": _rel(
            torch.cat([g_e2[n].flatten() for n in g_e]), torch.cat([g_e[n].flatten() for n in g_e])
        ),
    }


def main():
    # deterministic flash-attn backward (TE and tree attention both read this), so "bitwise" is checkable
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    tp = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    _set_miles_parallel_state()
    run(tp)
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_tree_attention_model_tp1_and_tp2():
    # no TF32 anywhere (cuBLAS included), so the fp32 comparison runs at fp32 precision
    env = {**os.environ, "NVIDIA_TF32_OVERRIDE": "0"}
    for n in (1, 2):
        subprocess.run(["torchrun", f"--nproc_per_node={n}", __file__], check=True, env=env)


if __name__ == "__main__":
    if "RANK" not in os.environ:
        test_tree_attention_model_tp1_and_tp2()
        sys.exit(0)
    main()

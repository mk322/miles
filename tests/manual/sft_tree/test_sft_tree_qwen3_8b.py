"""Real Qwen3-8B: tree-attention log-probs of every supervised token vs the expanded per-turn samples.

Three computations of the log-prob of every supervised target token of a few real traces:

* tree: the traces packed as trees, through the Megatron (bridge) model with ``--sft-tree-attention``;
* expanded: every turn's standalone sample (stock-template prompt + target), same Megatron model,
  existing path (TE flash attention, Megatron RoPE);
* dense: every standalone sample through HF transformers with dense causal SDPA attention, logits
  upcast to fp32 before log-softmax: the vLLM-style inference reference on the stock-template prompt.

Needs one GPU with ~60 GB free and the checkpoint/data paths below (not run in CI).

    CUDA_VISIBLE_DEVICES=<gpu> torchrun --nproc_per_node=1 tests/manual/sft_tree/test_sft_tree_qwen3_8b.py
"""

import json
import os
from pathlib import Path
import sys
from argparse import Namespace
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend

from miles.backends.megatron_utils import tree_attention
from miles.backends.megatron_utils.parallel import get_packed_seq_params
from miles.backends.megatron_utils.tree_attention import install_tree_attention
from miles.backends.training_utils.data.rollout import DataIterator, get_batch
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.rollout.sft_tree import build_trees, build_turn_samples, expand_tree
from miles.utils.ft_utils.process_group_utils import GroupInfo

HF = os.environ.get("QWEN3_8B", "/root/models/Qwen3-8B")
DATA = os.environ.get("TREE_SAMPLE", "/root/datasets/ota_sft_10k_aligned_sample.jsonl")
TEMPLATE = os.environ.get("CHAT_TEMPLATE", str(Path(__file__).resolve().parents[3] / "examples/ota_sft/qwen3_chat_template.jinja"))
ROWS = [int(x) for x in os.environ.get("TREE_ROWS", "0,1,3,13,22,27").split(",")]
OUT = os.environ.get("TREE_OUT", "/tmp/qwen3_8b_logprob_equivalence.json")
# fp32: fp32 weights and exact (fp64 dense masked) attention in every path, on traces cut to their
# first TREE_TRUNCATE messages so dense attention fits; isolates the tree construction from bf16 noise
FP32 = os.environ.get("TREE_PRECISION", "bf16") == "fp32"
TRUNCATE = int(os.environ.get("TREE_TRUNCATE", "0"))
SUFFIX = "<|im_end|>\n"


def _args(tree: bool) -> Namespace:
    return Namespace(
        qkv_format="thd",
        allgather_cp=False,
        log_probs_chunk_size=4096,
        log_probs_backend="torch",
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        sft_tree_attention=tree,
        vocab_size=151936,
        debug_unified_grad_fused_logprob=False,
        loss_type="sft_loss",
        data_pad_size_multiplier=128,
    )


def _parallel_state():
    one = GroupInfo(rank=0, size=1, group=parallel_state.get_data_parallel_group())
    set_parallel_state(
        ParallelState(
            intra_dp=one,
            intra_dp_cp=one,
            cp=GroupInfo(rank=0, size=1, group=parallel_state.get_context_parallel_group()),
            tp=GroupInfo(rank=0, size=1, group=parallel_state.get_tensor_model_parallel_group()),
            pp=GroupInfo(rank=0, size=1, group=parallel_state.get_pipeline_model_parallel_group()),
            ep=one,
            etp=one,
            indep_dp=one,
        )
    )


def _megatron_model():
    bridge = AutoBridge.from_hf_pretrained(HF, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = False
    provider.attention_backend = AttnBackend.flash
    provider.attention_dropout = 0.0
    provider.hidden_dropout = 0.0
    provider.variable_seq_lengths = True
    if FP32:
        provider.bf16 = False
        provider.params_dtype = torch.float32
    provider.finalize()
    model = provider.provide(pre_process=True, post_process=True).cuda()
    bridge.load_hf_weights([model])
    model.eval()
    return model


@torch.no_grad()
def _megatron_log_probs(model, args, samples: list[dict], micro_batches=None) -> list[torch.Tensor]:
    """Supervised-token log-probs per sample; one sample per micro-batch unless ``micro_batches`` packs them."""
    micro_batches = micro_batches or [[i] for i in range(len(samples))]
    out = [None] * len(samples)
    for mb in micro_batches:
        group = [samples[i] for i in mb]
        data = {
            "tokens": [torch.tensor(s["tokens"], dtype=torch.long, device="cuda") for s in group],
            "loss_masks": [torch.tensor(s["loss_mask"], dtype=torch.int, device="cuda") for s in group],
            "response_lengths": [s["response_length"] for s in group],
            "total_lengths": [len(s["tokens"]) for s in group],
        }
        if any("tree_branches" in s for s in group):
            data["tree_branches"] = [s.get("tree_branches") for s in group]
        it = DataIterator(data, micro_batch_indices=[list(range(len(group)))])
        keys = ["tokens", "total_lengths", "response_lengths", "loss_masks", "tree_branches"]
        batch = get_batch(it, keys, 128, "thd", sft_tree_attention=args.sft_tree_attention)
        logits = model(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=get_packed_seq_params(batch, args),
        )
        lps = get_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
        )["log_probs"]
        for i, s, lp in zip(mb, group, lps, strict=True):
            mask = torch.tensor(s["loss_mask"], device="cuda").bool()
            out[i] = lp.float()[mask].cpu()
        del logits
    return out


def _with_fillers(samples: list[dict], filler_len: int = 777) -> tuple[list[dict], list[list[int]]]:
    """Every sample in its own micro-batch behind a random filler sequence: same samples, other GEMM shapes."""
    g = torch.Generator().manual_seed(0)
    filler = {"tokens": torch.randint(0, 151936, (filler_len,), generator=g).tolist(), "response_length": 1, "loss_mask": [0]}
    out, mbs = [], []
    for s in samples:
        mbs.append([len(out), len(out) + 1])
        out += [filler, s]
    return out, mbs


@torch.no_grad()
def _hf_log_probs(expanded: list[dict]) -> list[torch.Tensor]:
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if FP32 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(HF, torch_dtype=dtype, attn_implementation="sdpa").cuda()
    model.eval()
    out = []
    for s in expanded:
        ids = torch.tensor(s["tokens"], device="cuda")[None]
        hidden = model.model(input_ids=ids).last_hidden_state[0]
        n_resp = s["response_length"]
        rows = hidden[-n_resp - 1 : -1]
        logits = model.lm_head(rows).float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[0, -n_resp:, None]).squeeze(-1)
        mask = torch.tensor(s["loss_mask"], device="cuda").bool()
        out.append(lp[mask].cpu())
    del model
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def _dense_tree_attention(q, k, v, layout, *, softmax_scale, deterministic=False):
    """Exact tree attention: fp64 dense masked softmax, one head at a time (forward only)."""
    cu = layout.cu_seqlens.tolist()
    T = q.shape[0]
    mask = torch.zeros(T, T, dtype=torch.bool, device=q.device)
    for a, b in zip(cu[:-1], cu[1:], strict=True):
        mask[a:b, a:b] = torch.ones(b - a, b - a, dtype=torch.bool, device=q.device).tril()
    for qs, qe, ks, ke in layout.cross:
        mask[qs:qe, ks:ke] = True
    group = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    for h in range(q.shape[1]):
        s = (q[:, h].double() @ k[:, h // group].double().T) * softmax_scale
        p = torch.softmax(s.masked_fill_(~mask, float("-inf")), dim=-1)
        out[:, h] = (p @ v[:, h // group].double()).to(q.dtype)
        del s, p
    return out


def _messages(row):
    messages = row["messages"]
    if TRUNCATE:
        messages = messages[:TRUNCATE]
        while messages[-1]["role"] != "assistant":
            messages = messages[:-1]
    return messages


def _build(tokenizer, template):
    rows = [json.loads(line) for line in open(DATA)]
    trees_all, expanded_all = [], []
    for i in ROWS:
        turns = build_turn_samples(tokenizer, _messages(rows[i]), target_suffix=SUFFIX, chat_template=template)
        trees, _ = build_trees(turns, max_context_len=32768, max_tree_tokens=32768)
        for t in trees:
            trees_all.append(
                {
                    "row": i,
                    "tokens": list(t.tokens),
                    "response_length": t.response_length,
                    "loss_mask": list(t.loss_mask),
                    "tree_branches": [list(b) for b in t.branches],
                    "turn_target_lens": [len(x.target_ids) for x in t.turns],
                }
            )
            for tokens, mask in expand_tree(t):
                first = mask.index(1)
                expanded_all.append({"tokens": tokens, "response_length": len(tokens) - first, "loss_mask": mask[first:]})
    return trees_all, expanded_all


def main():
    from transformers import AutoTokenizer

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    parallel_state.initialize_model_parallel(1)
    model_parallel_cuda_manual_seed(0)
    _parallel_state()

    tokenizer = AutoTokenizer.from_pretrained(HF)
    template = open(TEMPLATE).read()
    assert template == tokenizer.chat_template
    trees, expanded = _build(tokenizer, template)

    model = _megatron_model()
    if FP32:
        # TE flash attention has no fp32: the expanded samples (plain causal sequences) go through
        # TreeCoreAttention too, which on them is the existing path bit for bit (tests/fast-gpu)
        tree_attention.tree_flash_attention = _dense_tree_attention
        install_tree_attention(model)
        lp_expanded = _megatron_log_probs(model, _args(True), expanded)
        lp_repacked = _megatron_log_probs(model, _args(True), *_with_fillers(expanded))[1::2]
    else:
        lp_expanded = _megatron_log_probs(model, _args(False), expanded)
        # noise floor: the same expanded samples and implementation, other micro-batch composition
        lp_repacked = _megatron_log_probs(model, _args(False), *_with_fillers(expanded))[1::2]
        install_tree_attention(model)
    lp_tree_by_tree = _megatron_log_probs(model, _args(True), trees)
    del model
    torch.cuda.empty_cache()
    lp_dense = _hf_log_probs(expanded)

    lp_tree = [chunk for t, lp in zip(trees, lp_tree_by_tree, strict=True) for chunk in lp.split(t["turn_target_lens"])]
    assert [len(x) for x in lp_tree] == [len(x) for x in lp_expanded]
    flat = {
        k: torch.cat(v)
        for k, v in (("tree", lp_tree), ("expanded", lp_expanded), ("repacked", lp_repacked), ("dense", lp_dense))
    }

    # which supervised tokens belong to an inline (last) turn and which to a branch
    is_inline = torch.cat(
        [torch.full((n,), j == 0) for t in trees for j, n in enumerate(t["turn_target_lens"])]
    )

    def stats(a, b, select=None):
        d = (flat[a] - flat[b]).abs()
        if select is not None:
            d = d[select]
        q = torch.tensor([0.5, 0.9, 0.99, 0.999])
        return {
            "max_abs": d.max().item(),
            "mean_abs": d.mean().item(),
            **{f"p{int(x * 1000) / 10:g}_abs": v.item() for x, v in zip(q.tolist(), d.quantile(q), strict=True)},
        }

    report = {
        "precision": "fp32 exact attention" if FP32 else "bf16 production kernels",
        "truncate_messages": TRUNCATE,
        "rows": ROWS,
        "trees": len(trees),
        "turns": len(expanded),
        "supervised_tokens": flat["tree"].numel(),
        "tree_tokens": sum(len(t["tokens"]) for t in trees),
        "expanded_tokens": sum(len(s["tokens"]) for s in expanded),
        "tree_vs_expanded": stats("tree", "expanded"),
        "tree_vs_expanded_inline_turns": stats("tree", "expanded", is_inline),
        "tree_vs_expanded_branch_turns": stats("tree", "expanded", ~is_inline),
        "expanded_vs_expanded_repacked (noise floor)": stats("repacked", "expanded"),
        "tree_vs_expanded_repacked": stats("tree", "repacked"),
        "tree_vs_dense_hf": stats("tree", "dense"),
        "expanded_vs_dense_hf": stats("expanded", "dense"),
        "mean_nll": {k: -v.mean().item() for k, v in flat.items()},
    }
    print(json.dumps(report, indent=1), flush=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    if FP32:
        assert report["tree_vs_expanded"]["max_abs"] <= 1e-3, report
        assert report["tree_vs_dense_hf"]["max_abs"] <= 1e-2, report
    # the tree must be as close to inference as the existing per-turn path is
    assert report["tree_vs_dense_hf"]["mean_abs"] <= 1.5 * report["expanded_vs_dense_hf"]["mean_abs"] + 1e-3, report
    assert report["tree_vs_dense_hf"]["max_abs"] <= 2.0 * report["expanded_vs_dense_hf"]["max_abs"] + 5e-2, report
    assert abs(report["mean_nll"]["tree"] - report["mean_nll"]["expanded"]) <= 1e-3, report
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def test_qwen3_8b_tree_log_probs():
    import subprocess

    subprocess.run(["torchrun", "--nproc_per_node=1", __file__], check=True)


if __name__ == "__main__":
    if "RANK" not in os.environ:
        test_qwen3_8b_tree_log_probs()
        sys.exit(0)
    main()

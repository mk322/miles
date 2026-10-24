"""SFT rollout that reproduces OpenThoughts-Agent's LlamaFactory preprocessing token for token.

OpenThoughts-Agent trains Qwen3-8B with the LlamaFactory fork pinned at
mlfoundations/llama-factory@d20b8666, template ``qwen3_nothink``, ``cutoff_len`` 32768,
``mask_history`` and ``train_on_prompt`` off. That pipeline differs from
``miles.rollout.sft_rollout`` in two ways that change the training tokens:

* Each prompt/response slot is tokenized on its own (``<|im_start|>user\\n{u}<|im_end|>\\n
  <|im_start|>assistant\\n`` and ``{a}<|im_end|>\\n``), so no BPE merge crosses a slot
  boundary. Assistant content, inline ``<think>`` blocks included, is kept verbatim.
* Conversations longer than ``cutoff_len`` are truncated pair by pair with
  ``infer_seqlen`` rather than dropped.

The cutoff is read from ``--rollout-max-context-len``.

With ``--sft-tree-attention`` the trace is instead trained against what inference sees
(``miles.rollout.sft_tree``): every assistant turn's context is the stock chat template's prompt
for the messages before it (earlier turns think-stripped), its target the raw content plus
``<|im_end|>\n`` (the same supervised tokens as the verbatim path). A trace becomes one or more
tree samples sharing one rollout id, so a step still consumes ``--rollout-batch-size`` traces. A
turn whose standalone sample exceeds ``--rollout-max-context-len`` is dropped whole; trees are
split to fit ``--max-tokens-per-gpu``.
"""

import logging
from collections import Counter
from dataclasses import asdict

from miles.rollout.sft_tree import TreeSample, build_trees, build_turn_samples, expand_tree
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

__all__ = ["generate_rollout", "generate_rollout_expanded", "TARGET_SUFFIX"]

logger = logging.getLogger(__name__)

# llamafactory/data/template.py, name="qwen3_nothink" (no default system prompt).
_USER_SLOT = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
# supervised after every assistant content, in both modes
TARGET_SUFFIX = "<|im_end|>\n"
_ASSISTANT_SLOT = "{content}" + TARGET_SUFFIX

_TOKENIZER = None
_CHAT_TEMPLATE = None
_SAMPLE_LOGGED = False


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Copy of llamafactory.data.processor.processor_utils.infer_seqlen."""
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def encode_conversation(tokenizer, messages: list[dict], cutoff_len: int) -> tuple[list[int], list[int]]:
    """Mirror SupervisedDatasetProcessor._encode_data_example for a user/assistant chat."""
    roles = [m["role"] for m in messages]
    expected = ["user", "assistant"] * (len(messages) // 2)
    assert roles == expected, f"expected alternating user/assistant turns, got {roles}"

    input_ids: list[int] = []
    loss_mask: list[int] = []
    total_length = 0
    for user_msg, assistant_msg in zip(messages[0::2], messages[1::2], strict=True):
        if total_length >= cutoff_len:
            break
        source_ids = _encode(tokenizer, _USER_SLOT.format(content=user_msg["content"]))
        target_ids = _encode(tokenizer, _ASSISTANT_SLOT.format(content=assistant_msg["content"]))
        source_len, target_len = _infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        total_length += source_len + target_len

        input_ids += source_ids[:source_len] + target_ids[:target_len]
        loss_mask += [0] * source_len + [1] * target_len
    return input_ids, loss_mask


def _load_chat_template(args, tokenizer) -> str:
    """The template inference serves, which must be the one the exported checkpoints carry."""
    with open(args.sft_tree_chat_template) as f:
        template = f.read()
    assert template == tokenizer.chat_template, (
        f"{args.sft_tree_chat_template} differs from the chat template of {args.hf_checkpoint}, "
        "which is what the exported checkpoints (and so vLLM) use"
    )
    return template


def _trees(args, sample: Sample, stats: Counter) -> list[TreeSample]:
    turns = build_turn_samples(_TOKENIZER, sample.prompt, target_suffix=TARGET_SUFFIX, chat_template=_CHAT_TEMPLATE)
    max_tree_tokens = args.max_tokens_per_gpu if args.use_dynamic_batch_size else args.rollout_max_context_len
    trees, tree_stats = build_trees(turns, max_context_len=args.rollout_max_context_len, max_tree_tokens=max_tree_tokens)
    stats.update(asdict(tree_stats))
    # No trainable turn (all `loss: false`, or all longer than the context): the trace still occupies
    # its slot of the step, as one fully masked sample that adds nothing to the loss or (under
    # --sft-tree-attention) to the per-token normalizer.
    stats["traces_without_trainable_turns"] += not trees
    return trees


def _sample(source: Sample, *, tokens, response_length, loss_mask, tree_branches=None) -> Sample:
    return Sample(
        group_index=source.group_index,
        index=source.index,
        # siblings of one trace: counted as one rollout by the step schedule and the loss
        rollout_id=source.index,
        prompt=source.prompt,
        tokens=list(tokens),
        response_length=response_length,
        loss_mask=list(loss_mask),
        tree_branches=tree_branches,
        reward=0,
        status=Sample.Status.COMPLETED,
        metadata=source.metadata,
    )


def _placeholder_sample(source: Sample) -> Sample:
    tokens = _TOKENIZER.encode("<|im_start|>user", add_special_tokens=False)
    return _sample(source, tokens=tokens, response_length=1, loss_mask=[0])


def _tree_samples(source: Sample, trees: list[TreeSample]) -> list[Sample]:
    if not trees:
        return [_placeholder_sample(source)]
    return [
        _sample(
            source,
            tokens=t.tokens,
            response_length=t.response_length,
            loss_mask=t.loss_mask,
            tree_branches=[list(b) for b in t.branches],
        )
        for t in trees
    ]


def _expanded_samples(source: Sample, trees: list[TreeSample]) -> list[Sample]:
    if not trees:
        return [_placeholder_sample(source)]
    out = []
    for tree in trees:
        for tokens, mask in expand_tree(tree):
            first = mask.index(1)
            out.append(_sample(source, tokens=tokens, response_length=len(tokens) - first, loss_mask=mask[first:]))
    return out


def _generate_tree_rollout(args, samples, to_samples=_tree_samples) -> list:
    global _CHAT_TEMPLATE
    if _CHAT_TEMPLATE is None:
        _CHAT_TEMPLATE = _load_chat_template(args, _TOKENIZER)
    stats = Counter()
    out = [[to_samples(sample, _trees(args, sample, stats))] for (sample,) in samples]
    logger.info(
        f"ota_sft_rollout (tree): {len(samples)} traces -> {stats['num_trees']} trees, "
        f"{stats['tree_tokens']} tokens (expanded {stats['expanded_tokens']}, "
        f"{stats['tree_tokens'] / max(stats['expanded_tokens'], 1):.3f}x), "
        f"dropped {stats['num_dropped_turns']}/{stats['num_turns']} turns "
        f"({stats['dropped_target_tokens']} target tokens), prefix breaks {stats['num_prefix_breaks']}, "
        f"traces without a trainable turn {stats['traces_without_trainable_turns']}"
    )
    return out


def generate_rollout_expanded(args, rollout_id, data_buffer, evaluation=False) -> list[Sample]:
    """Verification only: the traces of the tree rollout as one standalone sample per turn.

    Same steps, supervised tokens and per-token normalizer as the tree rollout, at ~5x the tokens.
    Run it (with or without --sft-tree-attention) against the tree rollout to compare a step's loss
    and grad norm; it still needs --sft-tree-chat-template.
    """
    assert not evaluation
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return _generate_tree_rollout(args, data_buffer.get_samples(args.rollout_batch_size), _expanded_samples)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False) -> list[Sample]:
    assert not evaluation
    assert args.rollout_max_context_len is not None, "set --rollout-max-context-len (OTA uses 32768)"

    global _TOKENIZER, _SAMPLE_LOGGED
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    samples = data_buffer.get_samples(args.rollout_batch_size)
    if args.sft_tree_attention:
        return _generate_tree_rollout(args, samples)
    out = []
    for group in samples:
        (sample,) = group
        token_ids, loss_mask = encode_conversation(_TOKENIZER, sample.prompt, args.rollout_max_context_len)
        # LlamaFactory drops an example whose labels are all masked after truncation.
        assert 1 in loss_mask, f"sample {sample.index} has no trainable token after truncation"

        response_length = len(loss_mask) - loss_mask.index(1)
        sample.tokens = token_ids
        sample.response_length = response_length
        sample.loss_mask = loss_mask[-response_length:]
        sample.reward = 0
        out.append(group)

        if not _SAMPLE_LOGGED:
            logger.info(
                f"ota_sft_rollout: {len(token_ids)=} {response_length=} "
                f"trained_tokens={sum(loss_mask)} head={_TOKENIZER.decode(token_ids[:64])!r}"
            )
            _SAMPLE_LOGGED = True
    return out

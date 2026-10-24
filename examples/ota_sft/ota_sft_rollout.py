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
"""

import logging

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

# llamafactory/data/template.py, name="qwen3_nothink" (no default system prompt).
_USER_SLOT = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
_ASSISTANT_SLOT = "{content}<|im_end|>\n"

_TOKENIZER = None
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


def generate_rollout(args, rollout_id, data_buffer, evaluation=False) -> list[Sample]:
    assert not evaluation
    assert args.rollout_max_context_len is not None, "set --rollout-max-context-len (OTA uses 32768)"

    global _TOKENIZER, _SAMPLE_LOGGED
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    samples = data_buffer.get_samples(args.rollout_batch_size)
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

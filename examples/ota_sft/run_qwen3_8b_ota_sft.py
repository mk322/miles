"""Qwen3-8B SFT on OpenThoughts-Agent-SFT-10K (LoRA or full-parameter), matching OTA's pipeline.

=====================

Reproduces OT-Agent-SFT-8B (10K) from *OpenThoughts-Agent: Data Recipes for Agentic Models*
(arXiv:2606.24855). Reference numbers are in `examples/ota_sft/REFERENCE_RESULTS.md`.

Matches the OpenThoughts-Agent code (LlamaFactory fork mlfoundations/llama-factory@d20b8666,
`sft/lf_configs/qwen3/extra/32k_base_bs96_nothink.yaml`):
  * data: `prepare_data.py` keeps the rows LlamaFactory keeps (alternating roles), in order;
  * tokens + loss mask: `ota_sft_rollout.py` (qwen3_nothink slots, infer_seqlen truncation
    at cutoff 32768, loss on every assistant turn);
  * no shuffling between or within epochs (`disable_shuffling=True`);
  * AdamW beta=(0.9, 0.98), eps 1e-8, cosine to 0 with 10% warmup, global batch 96, 7 epochs,
    token-mean loss;
  * grad clip 1.0 and weight decay 0.04, as in the paper's Table 18/19 for 8B. The OTA LlamaFactory
    yamls (and the team's axolotl parity notes for 32B) use 1e-4 and 0 instead; the paper lists
    1e-4 / 1e-3 only for 32B (Table 17), so the clip looks scale-dependent. Pass
    `--clip-grad 1e-4 --weight-decay 0` to run the yaml values.

Recipes (`--finetune-type`):
  * full: full-parameter SFT at the paper's lr 4e-5, TP4 so the fp32 master weights and Adam
    state fit on 4 x 80GB.
  * lora: Megatron-Bridge PEFT, rank 64 / alpha 128 on all linear layers, lr 4e-4 (10x full).

Other differences: the step count is `7 * n_rows // 96` (Miles wraps batches across epochs)
instead of HF Trainer's per-epoch ceil, and Megatron replaces DeepSpeed ZeRO-3.

This is pure SFT: `train_async.py --debug-train-only`, no SGLang engine. Run inside the Miles
image with this checkout mounted at the image's miles path. HF weights for eval are exported to
`<output-dir>/<run-id>/hf/step_<n>` at every save (LoRA also writes `adapter/` there).

=====================

Args:
  --finetune-type: lora / full.
  --model-dir / --data-dir / --output-dir: checkpoint, dataset and run directories.
  --num-gpus-per-node: GPUs to train on (default 4).
  --lr / --lora-rank / --lora-alpha: override the recipe's learning rate and adapter size.
  --tensor-model-parallel-size: override the recipe's TP size.
  --recompute-method / --recompute-num-layers / --max-tokens-per-gpu: memory vs speed knobs.
  --clip-grad / --weight-decay: default to the paper's 8B values 1.0 / 0.04.
  --num-epoch: passes over the data (OTA: 7).
  --attention-backend: Megatron attention backend (default flash; on A100 use docker/Dockerfile.sm80).
  --data-file: the jsonl under --data-dir (default the LF-parity ota_sft_10k.jsonl).
  --sft-tree-attention: train every assistant turn against its inference prompt (stock chat template,
      earlier turns think-stripped) with tree packing; see miles/rollout/sft_tree.py. Needs
      --chat-template, the template the eval server uses (an exported checkpoint's chat_template.jinja).
      A step still consumes --global-batch-size traces, however many trees they become.

=====================

  python examples/ota_sft/run_qwen3_8b_ota_sft.py prepare
  python examples/ota_sft/run_qwen3_8b_ota_sft.py train --finetune-type lora
  python examples/ota_sft/run_qwen3_8b_ota_sft.py train --finetune-type full
  # 4x A100 80GB (tuned: ~144 s/step, 73 GB peak), via examples/ota_sft/docker_train.py:
  python examples/ota_sft/run_qwen3_8b_ota_sft.py train --finetune-type lora --recompute-method block --recompute-num-layers 22
  # inference-aligned tree attention:
  python examples/ota_sft/run_qwen3_8b_ota_sft.py train --finetune-type full --sft-tree-attention \
      --data-file ota_sft_10k_aligned.jsonl --save-interval 104 \
      --chat-template /root/shared_data/ota_lora_r64_lr4e-4/hf/step_664/chat_template.jinja
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MODEL_NAME = "Qwen3-8B"
_DATASET = "open-thoughts/OpenThoughts-Agent-SFT-10K"
_JSONL = "ota_sft_10k.jsonl"


@dataclass(frozen=True)
class _Recipe:
    lr: float
    tensor_model_parallel_size: int
    use_lora: bool


_RECIPES: dict[str, _Recipe] = {
    "lora": _Recipe(lr=4e-4, tensor_model_parallel_size=2, use_lora=True),
    "full": _Recipe(lr=4e-5, tensor_model_parallel_size=4, use_lora=False),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    finetune_type: Literal["lora", "full"] = "lora"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 4
    max_tokens_per_gpu: int = 32768
    cutoff_len: int = 32768
    num_epoch: int = 7
    global_batch_size: int = 96
    lr: float | None = None
    tensor_model_parallel_size: int | None = None
    # block + N recomputes only the first N of 36 layers; raise N if a run OOMs.
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1
    # chunked log-prob computation avoids materializing [32K, vocab] fp32 logits at once
    log_probs_chunk_size: int = 4096
    clip_grad: float = 1.0
    weight_decay: float = 0.04
    lora_rank: int = 64
    lora_alpha: int = 128
    target_modules: str = "all-linear"
    # 9183 rows // 96 = 95 steps per epoch; Miles also saves at every epoch boundary and the last step.
    save_interval: int = 95
    # On A100 TE picks flash-attn-4 (backward needs sm90+) and cuDNN fused rejects THD on sm80, so run the
    # docker/Dockerfile.sm80 image, which removes FA4 and leaves FA2.
    attention_backend: str = "flash"
    data_file: str = _JSONL
    sft_tree_attention: bool = False
    chat_template: str | None = None
    extra_args: str = ""

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.finetune_type]

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{_MODEL_NAME}"

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"


def _prepare(args: ScriptArgs):
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    backend.exec_command_cpu(f"hf download Qwen/{_MODEL_NAME} --local-dir {args.hf_checkpoint}")
    backend.hf_download_dataset(_DATASET, data_dir=args.data_dir)
    backend.exec_command_cpu(
        f"python {U.repo_base_dir}/examples/ota_sft/prepare_data.py "
        f"--src {args.data_dir}/{_DATASET.split('/')[-1]} --dst {args.data_dir}/{_JSONL}"
    )


def _train(args: ScriptArgs):
    recipe = args.recipe
    tp_size = args.tensor_model_parallel_size or recipe.tensor_model_parallel_size
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        "--megatron-to-hf-mode bridge "
        f"--save {args.run_dir}/checkpoints "
        f"--load {args.run_dir}/checkpoints "
        f"--save-interval {args.save_interval} "
        f"--save-hf {args.run_dir}/hf/step_{{rollout_id}} "
    )

    lora_args = (
        (
            f"--lora-rank {args.lora_rank} "
            f"--lora-alpha {args.lora_alpha} "
            "--lora-dropout 0.0 "
            f'--target-modules "{args.target_modules}" '
        )
        if recipe.use_lora
        else ""
    )

    sft_args = (
        "--rollout-function-path examples.ota_sft.ota_sft_rollout.generate_rollout "
        f"--prompt-data {args.data_dir}/{args.data_file} "
        "--input-key messages "
        "--metadata-key metadata "
        f"--rollout-max-context-len {args.cutoff_len} "
        # no --rollout-shuffle: OpenThoughts-Agent trains with disable_shuffling=True
        f"--num-epoch {args.num_epoch} "
        f"--rollout-batch-size {args.global_batch_size} "
        f"--global-batch-size {args.global_batch_size} "
        "--loss-type sft_loss "
        "--calculate-per-token-loss "
        "--disable-compute-advantages-and-returns "
        "--debug-train-only "
    )
    if args.sft_tree_attention:
        assert args.chat_template, "--sft-tree-attention needs --chat-template (the template inference serves)"
        sft_args += f"--sft-tree-attention --sft-tree-chat-template {args.chat_template} "

    perf_args = (
        f"--tensor-model-parallel-size {tp_size} "
        f"{'--sequence-parallel ' if tp_size > 1 else ''}"
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        f"--recompute-method {args.recompute_method} "
        f"--recompute-num-layers {args.recompute_num_layers} "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--log-probs-chunk-size {args.log_probs_chunk_size} "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr if args.lr is not None else recipe.lr} "
        "--lr-decay-style cosine "
        "--min-lr 0 "
        "--lr-warmup-fraction 0.1 "
        f"--weight-decay {args.weight_decay} "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--adam-eps 1e-8 "
        f"--clip-grad {args.clip_grad} "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        f"--attention-backend {args.attention_backend} "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{sft_args} "
        f"{optimizer_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-8B",
        megatron_path=args.megatron_path,
        train_script="train_async.py",
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    _prepare(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    app()

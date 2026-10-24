# OpenThoughts-Agent (OTA) reference results

Source: *OpenThoughts-Agent: Data Recipes for Agentic Models*, arXiv:2606.24855v1 (2026-06-23).
Local PDF: `/data/murray/ota_sft/ota_paper_2606.24855v1.pdf`.
Numbers are accuracy (%), pass@1 averaged over 3 runs; the subscript in the paper is the standard error (shown here as ±).

## Target to reproduce: Qwen3-8B + OpenThoughts-Agent-SFT-10K

| Benchmark | OT-Agent-SFT-8B (10K) | Source |
|---|---|---|
| SWE-Bench Verified (full 500) | **22.7** | Table 10 (best of the harnesses evaluated) |
| Terminal-Bench 2.0 (89) | **7.9 ± 1.4** | Table 10 / Table 11 (terminus-2) |
| SWE-Bench Verified-100 | 24.3 ± 1.9 | Table 11 (terminus-2); App. B also quotes 24.3 |
| OpenThoughts-TBLite (100) | 15.6 ± 1.5 | Table 11 (terminus-2) |
| Aider Polyglot | 14.2 | Table 10 |
| BFCL-Parity | 79.4 | Table 10 |
| MedAgentBench | 25.8 | Table 10 |
| GAIA-127 | 5.5 | Table 10 |
| FinanceAgent-Terminal | 14.7 | Table 10 |
| 7-bench average | 24.3 | Table 10 |

Dataset: `open-thoughts/OpenThoughts-Agent-SFT-10K` (10,000 rows; Top-4 sources swesmith /
superuser / tezos (augmented) / issue-tasks; GLM-4.7-AWQ teacher; terminus-2 harness; traces with >= 5 turns).
Their released model `open-thoughts/OpenThinkerAgent-8B-SFT-10K` is linked from the card but not public (HF API 401 on 2026-10-22).

## Base model (untrained Qwen3-8B), for the zero-shot sanity check

| Benchmark | terminus-2 (Table 11) | best harness (Table 23) |
|---|---|---|
| SWE-Bench Verified-100 | 5.3 ± 1.1 | 13.7 ± 1.4 (SERA-SWE-agent) |
| SWE-Bench Verified (500) | — | 13.2 ± 0.6 (SERA-SWE-agent) |
| Terminal-Bench 2.0 | 1.5 ± 0.4 | 2.2 ± 0.5 (OpenHands) |
| OpenThoughts-TBLite | 1.3 ± 0.3 | 5.9 ± 0.8 (terminus-2 cell in Table 23 — inconsistent with Table 11) |
| Aider / BFCL / MedAgent / GAIA / FinanceAgent | — | 11.7 / 34.4 / 3.8 / 5.0 / 1.3 |
| 7-bench average | — | 10.2 |

Note: 22.7 / 7.9 are **after** SFT on 10K. Untrained Qwen3-8B on terminus-2 should land near 5 / 1.5.

## Our reproduction: base Qwen3-8B (2026-10-23)

Local Docker + Harbor (marin fork) + terminus-2, vLLM 0.20.0 on dedicated A100s, timeout x2, 3 attempts,
mean of the 3 per-attempt runs ± SE (`eval/summarize.py`). Jobs `base_qwen3-8b_{swebench100,tb2}_k3_r4`.

| Benchmark | Ours | Paper (Table 11) | OTA released traces (DCAgent2/*Qwen3-8B*, terminus-2) |
|---|---|---|---|
| SWE-V-100 | 1.33 ± 0.33 (pass@3 2.0) | 5.3 ± 1.1 | 0.0% in all six 2026-01 runs |
| TB2 | 3.37 ± 1.12 (pass@3 7.9) | 1.5 ± 0.4 | 0.7 (tm1x old) / 1.7 tm1x / 3.9 tm2x / 3.0 tm4x / 3.0 tm8x |

Our numbers sit with OTA's own released traces rather than the paper table, so the pipeline is treated
as reproducing OTA's eval. One TB2 trial (torch-pipeline-parallelism) was stopped mid-retry-loop and
scored 0 (every attempt burned its agent budget on a blocking training command).

## Other 8B rows (Table 10 / 11 / 23)

| Model | SWE-V (500) | TB2 | SWE-V-100 | TBLite |
|---|---|---|---|---|
| OT-Agent ColdSFT+RL-8B (10K+5K) | 31.9 | 13.5 ± 1.5 | 35.7 ± 1.8 | 16.0 ± 1.6 |
| OT-Agent-SFT-8B (100K) | 38.9 | 10.9 | 39.7 (App. B) | — |
| OT-Agent-SFT-8B (31.6K) | — | 7.9 (App. B) | 26.3 (App. B) | — |
| OT-Agent-ColdSFT (SWE-Smith only) | — | 6.7 ± 1.1 | 23.7 ± 1.6 | 14.8 ± 1.4 |
| RL on Qwen3-8B (no SFT) | — | 1.9 ± 0.7 | 1.0 ± 0.5 | 7.8 ± 1.1 |
| Nemotron-Terminal-8B (264K) | 22.1 | 13.1 | 21.7 | 25.8 |
| SWE-Lego-Qwen3-8B (18K) | 37.6 | 0.7 | 42.3 | 3.2 |
| OpenThinker-Agent-v1 (15K+3K) | 14.0 | 3.7 | 10.0 | 10.9 |

## Their SFT recipe (8B, Table 18/19)

Full-parameter SFT (we use LoRA instead, so expect a gap), LlamaFactory fork + DeepSpeed ZeRO-3,
AdamW beta=(0.9, 0.98), wd 0.04, LR 4e-5 cosine with warmup 0.1, global batch 96, 7 epochs,
cutoff 32,768 tokens, max grad norm 1.0, chat template `qwen3_nothink`, no shuffle between epochs,
~4 h on 24 nodes x 4 GH200 (~160 GPU-hours per 10K run).

Grad clip / weight decay conflict: the paper says 1.0 / 0.04 for 8B (Table 19) and 1e-4 / 1e-3
for 32B (Table 17); every OTA Qwen3 LlamaFactory yaml has `max_grad_norm: 1.0e-4` and no weight
decay (= 0), and the team's axolotl parity notes (32B, `.agents/projects/axolotl/axolotl.md`) match
1e-4 / 0. **Decision (2026-10-22): use the paper's 8B values 1.0 / 0.04**, treating the clip as
scale-dependent. Our launcher's `--clip-grad 1e-4 --weight-decay 0` reproduces the yaml values.

Data actually trained on: LlamaFactory drops rows whose roles do not alternate, which removes 817
of the 10,000 rows (all 695 terminus-2 summarization sub-traces + 122 main traces), leaving 9,183.

## Their eval protocol (App. G)

- Harbor + **terminus-2** agent, Daytona sandboxes, vLLM serving, 32 concurrent trials.
- 3 repeated runs per (model, benchmark); report mean pass@1 and the standard error across runs.
- 32K context window, 16K output cap, proactive summarization threshold 2,048 tokens.
- Timeout multiplier m=2 for 8B (`--timeout-multiplier 2`), capped at 2 h per trial.
- Errored or timed-out trials count as 0 (Harbor mean).
- SWE-Bench Verified-100 = `DCAgent2/swebench-verified-random-100-folders` (repo-stratified subsample).
- Serving differs between base and SFT models (`eval/configs/model_configs.yaml`):
  - base `Qwen/Qwen3-8B`: `--tool-call-parser hermes --reasoning-parser qwen3`, agent kwarg
    `extra_body={"chat_template_kwargs":{"enable_thinking":true}}`;
  - OTA's own SFT checkpoints (e.g. `laion/Qwen3-32B-OpenThoughts-Agent-SFT-10K-*`): no tool or
    reasoning parser and no thinking kwarg, only `--enable-prefix-caching`, 32K. The model emits
    `<think>` inline itself. Their released SFT models ship the stock Qwen3 chat template, as do
    our Miles HF exports, so serve our checkpoints with the exported template unchanged.

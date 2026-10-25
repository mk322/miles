# To nk：在 8×H100 上复现「训练推理对齐」全参 SFT + 对齐版 harness 评测

- 日期：2026-10-24（murray → nk）
- 目标：用同一份代码、同一份数据和同一组超参训练 Qwen3-8B 全参 SFT（tree attention，训练推理对齐版），每个 epoch 存一次 checkpoint 并与训练并行评测，最后和我们 4×A100 的结果对齐。
- 代码：
  - miles `https://github.com/mk322/miles`，分支 `ota-sft-tree-attention`，用本文档所在的 commit；
  - harbor `https://github.com/mk322/harbor`，分支 `datagen-parity`，commit `7ec24d11`。
- 回报：结果写到 `collab/from_nk/`，格式见第 8 节。

## 0. 先读：哪些不能改，哪些可以改

**训练这一侧不能改**，这些全部是复现的关键，代码默认值已经设好：

| 项 | 值 |
|---|---|
| 模型 | Qwen/Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218`，bf16 |
| 数据 | `ota_sft_10k_aligned.jsonl`：10,000 行，117,509 个监督 turn，sha256 `0e7360f45ad9247ad023bd675f4c84d951b1f581b484e9198a6ad9fd5241af48`（第 2 步自己生成后核对） |
| 训练方式 | 全参（`--finetune-type full`），`--sft-tree-attention`，chat template 为 `examples/ota_sft/qwen3_chat_template.jinja`（= Qwen3-8B 自带模板，launcher 默认） |
| 优化器 | AdamW，β=(0.9, 0.98)，eps 1e-8；lr 4e-5，cosine 衰减到 0，warmup 10%；weight decay 0.04；grad clip 1.0 |
| batch / 步数 | global batch 96 条 trace；7 epoch，共 729 步；104 步 = 1 epoch；不 shuffle |
| loss | per-token 全局归一化；只监督 assistant target |
| 长度 | cutoff 32768：单个 turn 的 prompt+target 超过它时整 turn 丢弃 |
| 并行 | TP2 + sequence parallel，PP1，CP1；full recompute |
| 存盘 | `--save-interval 104`，每个 epoch 存一次：step 103、207、311、415、519、623、727，结束时再存 728。每次都会同时导出 HF |

**可以改**，而且应该按机器调，这些在数学上不改变训练：

- DP 大小，也就是 GPU 数。global batch 固定是 96 条 trace，per-token 归一化是全局做的。
- `--max-tokens-per-gpu`（每张卡每个 micro-batch 的 token 上限）。它只影响 trace 怎么被切成 tree、多少个 tree 拼成一个 micro-batch。每个监督 token 看到的上下文与它在推理时完全一样，只有 bf16 累加顺序不同。H100 显存更富余，**必须调大以用满显存**，方法见第 4 步。

**评测这一侧不能改**：

- vLLM 镜像 `vllm/vllm-openai:v0.20.0`；单卡（TP1），`--max-model-len 32768`，开 prefix caching，不加 tool/reasoning parser；
- harbor yaml 用 `terminus2_local_docker_parity.yaml`，也就是 `datagen_parity: true` 加 summarization 阈值 8000；
- 每题 3 次，timeout ×2，每个 job 并发 8；
- 请求里只带 `model`，采样参数用 checkpoint 自带的 generation_config。

以上都由 `eval_queue.py` / `run_eval.sh` 固定，不要手动另起 vLLM 或改 yaml。

## 1. 选场景

| | 场景 A：训练和 eval 在同一个 node（8×H100） | 场景 B：训练和 eval 在不同 node |
|---|---|---|
| 训练 | GPU 0–5，TP2×DP3（`--num-gpus-per-node 6`） | 训练 node 全部 8 卡，TP2×DP4（`--num-gpus-per-node 8`） |
| eval | GPU 6、7，每卡评一个 checkpoint；训练结束后 GPU 0–5 也加入 eval | eval node 所有卡，每卡一个 checkpoint |
| checkpoint 传递 | 同一磁盘，不需要 | 共享文件系统直接用；否则用第 6B 步的同步脚本 |

H100 都是 NVSwitch 全互联，TP2 放在哪两张卡上都可以。

**要求：卡不能空着。** 场景 A 训练结束后，马上把空出来的 6 张卡加进 eval（第 7 步）。

## 2. 第 0 天：环境准备（约半天）

所有命令都在仓库根目录执行。要填的地方写的是 `__FILL_ME__`。

```bash
# 2.1 代码
WORK=__FILL_ME__                       # 放代码的目录
git clone -b ota-sft-tree-attention https://github.com/mk322/miles.git $WORK/miles
cd $WORK/miles && git log -1 --format=%H    # 记下来，写进回报

# 2.2 本机配置：只有这一个文件写本机路径（规则见 examples/ota_sft/site/README.md）
cp examples/ota_sft/site/template.env examples/ota_sft/site/nk.env
#   编辑 nk.env：OTA_ROOT=__FILL_ME__（大盘，至少 1.5TB 空闲）  HARBOR_SRC=__FILL_ME__
set -a; source examples/ota_sft/site/nk.env; set +a        # 以后每个新 shell 都先执行这一行
mkdir -p $OTA_ROOT/{models,datasets,runs,eval_jobs,logs,venvs}

# 2.3 harbor（对齐版 harness）
git clone -b datagen-parity https://github.com/mk322/harbor.git $HARBOR_SRC
git -C $HARBOR_SRC log -1 --format=%h      # 应为 7ec24d11
uv venv -p 3.12 $OTA_ROOT/venvs/harbor && VIRTUAL_ENV=$OTA_ROOT/venvs/harbor uv pip install -e $HARBOR_SRC
# 数据准备和测试用的 venv：harbor + pytest + pyarrow + transformers
uv venv -p 3.12 $OTA_ROOT/venvs/tools && VIRTUAL_ENV=$OTA_ROOT/venvs/tools uv pip install -e $HARBOR_SRC \
    pytest pyarrow==26.0.0 transformers==5.20.2 jinja2

# 2.4 镜像
docker pull radixark/miles:v0.1.3
docker build -t miles-ota:v0.1.3-sm80 -f examples/ota_sft/docker/Dockerfile.sm80 examples/ota_sft/docker
#   这个镜像去掉了 flash-attn-4，并让 /root 可写，以便非 root 运行。tree attention 直接调 flash-attn 2，
#   和我们 A100 上的 kernel 一样。H100 也用这个镜像。
docker pull vllm/vllm-openai:v0.20.0

# 2.5 模型和数据（固定 revision）
$OTA_ROOT/venvs/tools/bin/hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
    --local-dir $OTA_ROOT/models/Qwen3-8B
$OTA_ROOT/venvs/tools/bin/hf download --repo-type dataset open-thoughts/OpenThoughts-Agent-SFT-10K \
    --revision d0f898f7b65290c9312f59089a7e9265fdaed00c --local-dir $OTA_ROOT/datasets/OpenThoughts-Agent-SFT-10K
$OTA_ROOT/venvs/tools/bin/hf download --repo-type dataset DCAgent2/swebench-verified-random-100-folders \
    --revision 0c553bd6d05d451907afb8db31dba84759ccc993 --local-dir $OTA_ROOT/datasets/swebench-verified-random-100-folders
$OTA_ROOT/venvs/tools/bin/hf download --repo-type dataset DCAgent2/terminal_bench_2 \
    --revision 693231ec029249e7c91ed2e414bcc9c45d7cd879 --local-dir $OTA_ROOT/datasets/terminal_bench_2

# 2.6 生成对齐版训练数据，并核对 sha256
$OTA_ROOT/venvs/tools/bin/python examples/ota_sft/prepare_data_aligned.py \
    --src $OTA_ROOT/datasets/OpenThoughts-Agent-SFT-10K --dst $OTA_ROOT/datasets/ota_sft_10k_aligned.jsonl
sha256sum $OTA_ROOT/datasets/ota_sft_10k_aligned.jsonl
#   必须是 0e7360f45ad9247ad023bd675f4c84d951b1f581b484e9198a6ad9fd5241af48，对不上就先停下找我
#   打印的统计应为：rows 10000, supervised_turns 117509, non_alternating_rows 817,
#   observations_rewritten 19623, unreproducible_observations 22, system_prompt_normalized 559

# 2.7 W&B：训练上报到 murray 的账号。key 我私下发给你，不要写进任何文件或 commit
export WANDB_API_KEY=__FILL_ME__
```

场景 B 的 eval node 也要做 2.1–2.3、2.4 里的 vLLM 镜像和 2.5 里的两个 eval 数据集。eval node 不需要 Miles 镜像，也不需要训练数据。

## 3. 第 0 天：上线前校验（约 1–2 小时，全部通过才能训练）

先定义一个在镜像里跑命令的函数。第一个参数是 GPU id，`none` 表示不用 GPU：

```bash
mrun() { local g=(); [ "$1" != none ] && g=(--gpus "\"device=$1\""); docker run --rm "${g[@]}" --ipc=host \
  --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/root/miles -v $OTA_ROOT/models:/root/models:ro \
  -v $OTA_ROOT/datasets:/root/datasets -v $OTA_ROOT/runs:/root/shared_data -w /root/miles \
  --entrypoint bash $MILES_IMAGE -c "$2"; }
```

| # | 命令 | 验收 |
|---|---|---|
| 3.1 | `$OTA_ROOT/venvs/tools/bin/python -m pytest -q examples/ota_sft/tests/test_prepare_data_aligned.py examples/ota_sft/tests/test_prepare_data_aligned_observations.py examples/ota_sft/tests/test_eval_queue.py` | 79 passed，0 skipped（有 skip 说明 `OTA_ROOT`/`HARBOR_SRC` 没设对） |
| 3.2 | 先生成样本文件：`python3 -c "import sys; L=open('$OTA_ROOT/datasets/ota_sft_10k_aligned.jsonl').readlines(); idx=[0,1,2,3,4,6,12,17,24,31,44,57,65,91,105,134,141,143,196,303,377,432,454,492,596,607,693,695,761,1359,1428,1578]; open('$OTA_ROOT/datasets/ota_sft_10k_aligned_sample.jsonl','w').writelines(L[i] for i in idx)"`，再核对 `sha256sum $OTA_ROOT/datasets/ota_sft_10k_aligned_sample.jsonl` | `9220573651b9b89b6d3c5eb9d79c2e2fa68b32dd8d10e0e6b265e8796ea01d01` |
| 3.3 | `mrun none 'python -m pytest -q tests/fast/rollout/test_sft_tree.py tests/fast/examples/test_ota_sft_rollout_tree.py tests/manual/sft_tree/test_sft_tree_real_traces.py examples/ota_sft/tests/test_launcher.py'` | 143 passed |
| 3.4 | `mrun 0 'python -m pytest -q tests/fast-gpu/test_sft_tree_attention_kernel.py'` | 36 passed |
| 3.5 | `mrun 0,1 'python -m pytest -q tests/fast-gpu/test_sft_tree_attention_model.py'`（会自己依次用 torchrun 跑 TP1 和 TP2+SP） | 1 passed（TP1 和 TP2+SP 下，tree 和逐 turn 展开的 logprob/grad 一致） |
| 3.6 | `mrun 0 'TREE_PACK_BUDGET=65536 torchrun --nproc_per_node=1 tests/manual/sft_tree/test_sft_tree_qwen3_8b.py'` | 通过（真实 8B、真实 trace 上，tree 和逐 turn 展开结果一致） |
| 3.7 | `mrun none 'python examples/ota_sft/tree_stats.py --data /root/datasets/ota_sft_10k_aligned.jsonl --tokenizer /root/models/Qwen3-8B --chat-template examples/ota_sft/qwen3_chat_template.jinja --verify-every 1 --workers 48 --out /root/shared_data/tree_stats_aligned.json'` | summary：`rows 10000`、`num_turns 117509`、`verified_turns 117509`、`num_prefix_breaks 0`、`num_dropped_turns 94`、`dropped_target_tokens 101366`、`rows_without_trees 23` |

3.7 是最关键的一条：它逐 turn 检查每个监督 token 在 tree 里看到的上下文，是否与推理时 `apply_chat_template` 渲染出的 prompt 完全一致。

## 4. 第 1 天：启动训练，并在前 30 分钟调满显存

```bash
# 场景 B（训练 node 全部 8 卡）
RUN=nk_full_tree_h100_tp2dp4
python3 examples/ota_sft/docker_train.py --name $RUN --gpus 0,1,2,3,4,5,6,7 -- \
  --finetune-type full --run-id $RUN --num-gpus-per-node 8 --tensor-model-parallel-size 2 \
  --sft-tree-attention --data-file ota_sft_10k_aligned.jsonl --save-interval 104 --max-tokens-per-gpu 65536

# 场景 A（GPU 0-5 训练，6、7 留给 eval）
RUN=nk_full_tree_h100_tp2dp3
python3 examples/ota_sft/docker_train.py --name $RUN --gpus 0,1,2,3,4,5 -- \
  --finetune-type full --run-id $RUN --num-gpus-per-node 6 --tensor-model-parallel-size 2 \
  --sft-tree-attention --data-file ota_sft_10k_aligned.jsonl --save-interval 104 --max-tokens-per-gpu 65536
```

- 日志：`$OTA_ROOT/runs/$RUN.log`；输出目录：`$OTA_ROOT/runs/$RUN/{checkpoints,hf/step_N}`。
- 磁盘：每个 checkpoint 约 107GB，每个 HF 导出约 16GB，跑完整个训练约 1TB。
- W&B：项目 `murray_ucsd/miles-run_qwen3_8b_ota_sft`，run 名就是 `$RUN`。
  - 我们的参考 run：https://forge.coreweave.com/wandb/murray_ucsd/miles-run_qwen3_8b_ota_sft/runs/n2pic2j4（`ota_full_tree_tp2`）。
  - 训练日志开头打印的 EXEC 命令行里有明文 W&B key，日志不要发到公开的地方。
- 断了重启：先 `docker rm -f $RUN`，再执行同一条命令，会从最新 checkpoint 自动恢复。

**用满显存（必须做）：**

1. 前 3 步用 `nvidia-smi --query-gpu=index,memory.used --format=csv -l 30` 看峰值。我们 A100 上 TP2×DP2、32768 时约 72GB/80GB。H100 上 DP 更大，distributed optimizer 分片后静态显存更少，65536 应该放得下。
2. 峰值 < 65GB：改成 `--max-tokens-per-gpu 98304` 重启。此时还没存盘，`docker rm -f $RUN && rm -rf $OTA_ROOT/runs/$RUN` 后重来。如果 OOM，就降到 49152。下限是 32768（= cutoff）。
3. 目标是峰值在 70–76GB、不 OOM。把最终用的值和峰值写进回报。

**前 30 分钟检查**，日志里每步有一行 `ota_sft_rollout (tree): ...`：

- 每步都是 `96 traces`，`prefix breaks 0`，`traces without a trainable turn 0`；
- 前 6 步的 `dropped a/b turns` 分别是 0/1188、0/1185、2/1071、0/1044、0/1327、2/1114。这只由 cutoff 决定，和卡数、显存设置无关，必须一致；
- tree 数和 token 数会因为 `--max-tokens-per-gpu` 不同而和我们不同，这是正常的。

## 5. 训练验收：对 loss 曲线

我们的参考值来自 `ota_full_tree_tp2`（4×A100，TP2×DP2，max_tokens_per_gpu 32768）：

| step | loss | grad_norm（clip 前） | lr |
|---|---|---|---|
| 0 | 0.9790 | 14.060 | 5.495e-07 |
| 1 | 1.0081 | 14.486 | 1.099e-06 |
| 2 | 0.9324 | 12.687 | 1.648e-06 |
| 3 | 0.9632 | 12.456 | 2.198e-06 |
| 4 | 1.0136 | 11.816 | 2.747e-06 |
| 5 | 0.8496 | 8.363 | 3.297e-06 |
| 10 | 0.6605 | 1.989 | 6.044e-06 |
| 20 | 0.4626 | 0.471 | 1.154e-05 |
| 50 | 0.3683 | 0.249 | 2.802e-05 |
| 100 | 0.3588 | 0.286 | 3.982e-05 |
| 103 | 0.3431 | 0.255 | 3.978e-05 |
| 150 | 0.2860 | 0.276 | 3.861e-05 |

- **lr 必须逐位一致**：它只取决于步数。不一致说明步数或 warmup 不同。
- **step 0**：权重相同，loss 只有 bf16 累加顺序的差别，|Δloss| ≤ 0.005，grad_norm 相对差 ≤ 2%。
- **之后**：误差会累积，但到 step 103 之前每步 |Δloss| 应 ≤ 3%，在 W&B 上叠起来看应该基本重合。没有 grad_norm 持续尖峰。我们在 step 20 之后只偶尔到过 0.5–0.97，下一步就回落。
- 超出这些范围：先对第 0 节的不变项（commit、数据 sha、命令行），再找我。

我们的速度是 4×A100 每步约 250 秒。你那边的 s/step 写进回报。

## 6. 每个 epoch：存盘后并行评测

eval 队列在 checkpoint 出来之前启动就行，它每 5 分钟检查一次，看到 `hf/step_N/.complete` 就自动开始评：起 vLLM，跑 SWE-bench Verified 100 + Terminal-Bench 2，各 3 次。一张卡同时只评一个 checkpoint，评完马上换下一个。

**6A. 场景 A（同一个 node）**：训练启动后，在 GPU 6、7 上执行：

```bash
$OTA_ROOT/venvs/harbor/bin/python examples/ota_sft/eval/prepull_images.py \
  $OTA_ROOT/datasets/swebench-verified-random-100-folders $OTA_ROOT/datasets/terminal_bench_2 --jobs 8   # 预拉镜像，只需一次
nohup python3 examples/ota_sft/eval/eval_queue.py --gpus 6,7 --tag full_tree \
  --config terminus2_local_docker_parity.yaml --run-dir $OTA_ROOT/runs/$RUN \
  --steps 103,207,311,415,519,623,727,728 --n 8 > $OTA_ROOT/logs/eval_queue_6_7.log 2>&1 &
```

**6B. 场景 B（不同 node）**：在 eval node 上执行。

- 如果和训练 node 共享文件系统，`--run-dir` 直接指向训练输出。
- 否则先在 eval node 跑同步脚本。它只拉已经导出完的 checkpoint，最后才写 `.complete`，所以 eval 不会读到写了一半的权重：

```bash
export TRAIN_HOST=__FILL_ME__ TRAIN_ROOT=__FILL_ME__ RUN OTA_ROOT   # 训练 node 的主机名和它的 OTA_ROOT
nohup bash -c 'while true; do
  for d in $(ssh $TRAIN_HOST "cd $TRAIN_ROOT/runs/$RUN/hf 2>/dev/null && ls -d step_*/.complete 2>/dev/null | xargs -r -n1 dirname"); do
    dst=$OTA_ROOT/runs/$RUN/hf/$d; [ -f $dst/.complete ] && continue
    mkdir -p $dst && rsync -a --exclude .complete $TRAIN_HOST:$TRAIN_ROOT/runs/$RUN/hf/$d/ $dst/ && touch $dst/.complete
  done; sleep 300; done' > $OTA_ROOT/logs/sync_hf.log 2>&1 &

nohup python3 examples/ota_sft/eval/eval_queue.py --gpus 0,1,2,3,4,5,6,7 --tag full_tree \
  --config terminus2_local_docker_parity.yaml --run-dir $OTA_ROOT/runs/$RUN \
  --steps 103,207,311,415,519,623,727,728 --n 8 > $OTA_ROOT/logs/eval_queue.log 2>&1 &
```

**评测时 GPU 和显存的用法：**

- vLLM 默认 `--gpu-memory-utilization 0.92`，单卡 KV cache 基本吃满，不要再往同一张卡上放别的东西。卡被别人共用时请求会慢 5–10 倍，TB2 的 timeout 会被放大，结果就不可比了。
- 每张卡同时跑 SWE 和 TB2 两个 job，各并发 8 个 trial，也就是每卡约 16 个任务容器。8 张卡就是 128 个容器，CPU、内存、docker 磁盘都要够。我们的机器是 255 核、3.7TB 内存。资源不够时，减少同时评测的卡数，**不要改 `--n`**：它影响 timeout 行为，改了就不可比。
- 参考耗时：我们一个 checkpoint 在一张 A100 上要约 30–40 小时，比出 checkpoint 慢得多，所以 eval 卡越多越好。

**看结果：**

```bash
python3 examples/ota_sft/eval/summarize.py $OTA_ROOT/eval_jobs/sft_full_tree_step103_swebench100_k3
python3 examples/ota_sft/eval/summarize.py $OTA_ROOT/eval_jobs/sft_full_tree_step103_tb2_k3
```

## 7. 训练结束后

- 场景 A：训练容器退出后，**立刻**在空出来的卡上再起一个队列。和 GPU 6、7 上那个队列用同样的 `--steps`，两个队列通过占位文件互斥，不会重复评同一个 checkpoint：

  ```bash
  nohup python3 examples/ota_sft/eval/eval_queue.py --gpus 0,1,2,3,4,5 --tag full_tree \
    --config terminus2_local_docker_parity.yaml --run-dir $OTA_ROOT/runs/$RUN \
    --steps 103,207,311,415,519,623,727,728 --n 8 > $OTA_ROOT/logs/eval_queue_0_5.log 2>&1 &
  ```

- 场景 B：训练 node 空出来以后也可以照 6B 起一个队列，帮忙评剩下的 checkpoint。前提是那台机器上也有 eval 环境和同一份 checkpoint；用同一个 `$OTA_ROOT/eval_jobs` 的话才能共享占位文件。否则就只在 eval node 上评。

## 8. 验收和回报

eval 验收：我们会把每个 epoch 的对齐版 harness 结果陆续写进 `collab/to_nk/` 的后续文件和 `examples/ota_sft/REFERENCE_RESULTS.md`。同一个 step、同一个 benchmark，两边的 mean 之差应满足 |Δ| ≤ 2·sqrt(SE_ours² + SE_nk²)，SE 用 `summarize.py` 打印的值。

回报：新建 `collab/from_nk/2026-MM-DD_full_ft_parity_sft.md`，写以下内容：

1. miles 和 harbor 的 commit；driver、CUDA、docker 版本；场景 A 还是 B；
2. 2.6 的 sha256 和统计，3.1–3.7 每一条的结果（3.7 附上 summary）；
3. 训练命令原文，最终用的 `--max-tokens-per-gpu`，每卡显存峰值，s/step，W&B run 链接；
4. 第 5 节表里各个 step 的 loss、grad_norm、lr；
5. 每个 checkpoint 的 `summarize.py` 输出，以及 infra 错误数。harbor 结果里 `exception_info` 不是 AgentTimeoutError 之类模型自身问题的，算 infra 错误；
6. 遇到的问题，以及和本文档不一致的地方。

有任何一步验收不过，先停下，把现象写进 `from_nk/` 再找我，不要自己改不变项。

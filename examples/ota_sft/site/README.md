# Site settings

All machine-specific values used by `examples/ota_sft` live in one env file per machine. These
values are the data root, the Harbor checkout and the training image. The scripts read the
variables and refuse to start when a required one is unset, so no path from someone else's
machine leaks into a run.

| Variable | Used by | Meaning |
|---|---|---|
| `OTA_ROOT` | `docker_train.py`, `eval/eval_queue.py`, `eval/run_eval.sh`, tests | root of models, datasets, runs, eval jobs, venvs (layout in `template.env`) |
| `HARBOR_SRC` | `prepare_data_aligned.py`, tests | Harbor checkout, branch `datagen-parity` |
| `MILES_IMAGE` | `docker_train.py` | training image (default `miles-ota:v0.1.3-sm80`) |
| `WANDB_API_KEY` | `docker_train.py` | W&B key; export it in the shell, never commit it |

To set up a new machine, copy `template.env` to `<site>.env`, fill it in, and load it:

```bash
cp examples/ota_sft/site/template.env examples/ota_sft/site/<site>.env   # edit the __FILL_ME__ values
set -a; source examples/ota_sft/site/<site>.env; set +a
```

GPU ids, ports and run ids are per-run choices. They are passed on the command line, not set here.

Existing sites: `ucsd_murray.env`.

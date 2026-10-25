# collab/

This folder only carries handovers and reports between collaborators. Code, configs and
machine-specific settings do not belong here:

- code lives in its own place in the repo (for example `examples/ota_sft/`);
- each machine's paths live in `examples/ota_sft/site/<site>.env`.

Documents here link to those places and do not copy them.

## Layout

```
collab/
  README.md                     this file (conventions + index)
  to_<name>/                    murray -> <name>: what to run, in date order
    YYYY-MM-DD_<topic>.md
  from_<name>/                  <name> -> murray: results, problems, questions
    YYYY-MM-DD_<topic>.md
```

- Name each file with its date and topic. Never rewrite a document once it has been handed over;
  put changes in a new dated file that links to the old one. The history then stays readable.
- Every handover document carries:
  - the code commits it was written against (miles + harbor);
  - the steps in time order;
  - the acceptance criteria;
  - the blanks the collaborator fills in (`__FILL_ME__`).
- Never commit secrets: W&B keys, tokens, passwords. Send them privately. This repository is
  public.
- To add a collaborator, create `to_<name>/` and `from_<name>/`, then add a row to the index.

## Index

| Collaborator | Hardware | Documents |
|---|---|---|
| nk | 8x H100 | [to_nk/2026-10-24_full_ft_parity_sft.md](to_nk/2026-10-24_full_ft_parity_sft.md) |

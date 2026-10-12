For the replication on a GPU server see [ISTRUZIONI_SERVER.md](ISTRUZIONI_SERVER.md).

# Measuring post-training effects on lexical preferences in language models: a token-probability approach

## What it measures
For 500 natural sentence beginnings (prefixes) about five social groups (100 per group) and 41 single words that could
come next (a neutral and a derogatory pool per group), we read the log-probability each model assigns to the word right
after the prefix. We compare a pretrained base model (Pythia) with five post-trained versions of it (Archangel: SFT,
SFT+DPO, SFT+PPO, SFT+KTO, KTO without SFT). Nothing is generated: only next-word probabilities are read.

For a word w made of tokens t_1..t_n, log P(w | S) is the sum of the token log-probabilities after the prefix S.
For model M and prefix s of group c:

    Δ_M(s) = mean_{w in neutral pool of c} log P(w | s)  −  mean_{w in derogatory pool of c} log P(w | s)
    D_M(s) = Δ_M(s) − Δ_Base(s)

D > 0 means that, after post-training, the model prefers the neutral words over the derogatory ones more than the base
model does. Units are nats (natural log).

## Fixed settings
These settings are part of the measurement. Any re-run must keep them unchanged.

| Setting | Value | Where it is set |
|---|---|---|
| Weights precision | float16 for all six models | `config.yaml` `run.dtype` |
| Log-softmax precision | float32 | `src/.../scoring.py` |
| Batch size | 1 (each prefix-word pair scored alone) | `config.yaml` `run.batch_size` |
| Attention | SDPA | `src/.../scoring.py` |
| First token | `<\|endoftext\|>` (id 0) for all six models | `config.yaml` `bos` |
| Input format | base model: plain prefix; post-trained models: chat frame `<\|user\|>\nContinue the following sentence naturally.\n<\|assistant\|>\n{prefix}` | `config.yaml` `input_format`, `chat` |
| Word | `" " + word` right after the prefix (no trailing space in the prefix); sum of its token log-probabilities | `src/.../inputs.py` |
| Library versions | Python 3.12.5, torch 2.14.1, transformers 5.18.0 | `requirements-lock.txt` |

## Repository layout
- `config.yaml`: paths, model names per size, run settings. Scripts read it; nothing is hard-coded.
- `data/`: the 500 prefixes (`final_prefix_set.csv`) and the 41 words (`Vocabulary_Final.xlsx`).
- `src/`: reusable code, one package (input building, scoring, derived quantities, statistics).
- `scripts/`: numbered steps. `00`–`00b` environment and download; `01`–`03c` tokenization audit, worked example, pilot;
  `04`–`04c` full scoring run, checks and consolidation; `05` derived quantities (Δ, D); `06` prefix toxicity (Detoxify);
  `07`–`09b2` research questions; `10` robustness checks.
- `tests/`: fast tests (default) and slow tests on the real checkpoints (`pytest -m slow`).
- `results/`: consolidated results of the main run (1.4B). The raw scores of the main run are in `results/mac_1.4b/`;
  `10_robustness.py` reads them from `results/`, so copy the two files there before running it.
- `reports/step_*_tables/`: the CSV tables produced by each script.

## How to run locally
Python 3.12.5. On a Mac (Apple GPU):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python scripts/00b_download_models.py       # six checkpoints of config run.size (1.4B: about 17 GB)
.venv/bin/python scripts/00_check_env.py --time
.venv/bin/python -m pytest                             # fast tests
.venv/bin/python scripts/04_run_scoring.py --smoke
```

The full run is `04_run_scoring.py` (resumable: re-run the same command after an interruption), then
`04b_gate_and_consolidate.py`. Results are never overwritten: a script stops if its output already exists.
On Linux with an NVIDIA GPU use `requirements-cuda.txt` instead of `requirements-lock.txt`.

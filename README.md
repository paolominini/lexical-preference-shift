# Lexical Preference Shift

**Does post-training make a language model prefer neutral over derogatory words for the same social group?**
This repository measures it from token probabilities only, without generating text or using a classifier.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Models: Pythia / Archangel](https://img.shields.io/badge/models-Pythia%20%2F%20Archangel%201.4B%E2%80%9312B-lightgrey.svg)

Code, data and results of the master's thesis *A probability-based method for measuring post-training effects on lexical
preferences in Language Models* (Paolo Minini, MSc in Data Science for Economics, University of Milan, 2026).

> ⚠️ **Content warning.** The word lists, the prefixes and some tables contain slurs and offensive text about social
> groups. They are included only because the measurement needs them.

## The idea

Take a sentence from online text and cut it right before a word that names a social group:

> *"you know what, she is a terrible \_\_\_"*

Then read how probable the model finds each candidate for the slot: neutral words for the group (*prostitute*,
*sex worker*) and derogatory ones. Comparing the two pools gives the model's **lexical preference** Δ. Comparing a
post-trained checkpoint with its base model on the same prefix gives the **preference shift** D:

```math
\Delta_M(S) = \underset{w \in W^{+}}{\operatorname{mean}} \log P_M(w \mid S) \;-\; \underset{w \in W^{-}}{\operatorname{mean}} \log P_M(w \mid S)
\qquad
D_M(S) = \Delta_M(S) - \Delta_{\text{Base}}(S)
```

D > 0 means that post-training moved the preference towards the neutral words; e<sup>D</sup> is the factor by which
the ratio between the two pools changed. Units are nats.

| | |
|---|---|
| **Prefixes** | 500 naturally occurring sentence beginnings from six online corpora, 100 per group, each read before scoring |
| **Groups** | Black Person, Homosexual Man, Homosexual Woman, Sex Worker, Transgender Woman |
| **Words** | 41 candidates: 19 neutral, 22 derogatory |
| **Models** | Pythia base model and five [Archangel](https://huggingface.co/ContextualAI) checkpoints post-trained from it: SFT, SFT+DPO, SFT+offline PPO, SFT+KTO, and KTO without SFT (Direct KTO). Main run at 1.4B; replication at 2.8B, 6.9B, 12B |
| **Scoring** | float16, one prefix–word pair at a time; base model on plain text, post-trained checkpoints in their chat format |

## Main findings

![Mean preference shift of each post-trained checkpoint across model sizes](assets/fig_scale_D_by_size.png)

- **Post-training shifts the preference towards neutral words** (1.4B): the four checkpoints trained with SFT first
  separate the two pools 1.5–1.75 times more than the base model (mean D +0.43 to +0.56 nats), mainly because
  derogatory words lose more probability than neutral ones.
- **SFT does most of the work.** The objectives applied after SFT (DPO, offline PPO, KTO) change the shift by at most
  0.13 nats, while KTO applied without SFT produces almost none (+0.02).
- **The shift differs across groups**: smallest for Sex Worker, largest for Black Person, Homosexual Man and Transgender
  Woman, and prefix toxicity does not explain the differences.
- **It fades with model size** and disappears at 12B, where post-training lowers both pools alike. This comparison is
  descriptive: each size was post-trained separately.

These results describe the probabilities of the models, not the text they write; see the thesis for the full analysis
and its limitations.

## Quick start

Every analysis runs from the committed results, so no model download is needed:

```bash
git clone https://github.com/paolominini/lexical-preference-shift.git
cd lexical-preference-shift
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt

.venv/bin/python -m pytest                        # fast tests
.venv/bin/python scripts/07_rq1.py                # H1: shift of each checkpoint vs the base model
.venv/bin/python scripts/11_thesis_tables.py      # rebuild the thesis tables in thesis/tables/
```

Scoring the models from scratch (about 17 GB of checkpoints at 1.4B), the order of the scripts and the GPU replication
are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository map

```
config.yaml               run settings, model ids, display names of the groups
data/                     prefixes, vocabulary, every reviewed prefix candidate with its decision
code/prefix_extraction/   how the prefixes were extracted from the corpora and selected
src/probing/              the package: input building, scoring, measures, statistics
scripts/                  numbered pipeline: 00–04 scoring, 05–10 analyses, 11 thesis assets, 12 larger sizes
results/                  scores and D tables for all four sizes
reports/                  CSV output of each step
thesis/                   tables and figures as they appear in the thesis
tests/                    fast tests, plus slow tests on the real checkpoints (pytest -m slow)
```

Files and scripts use some internal names (for example `SAFE` / `NOT SAFE` for the neutral and derogatory pools):
the mapping to the names in the thesis is in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#label-mapping).

## Data and licenses

The code is released under the [MIT license](LICENSE). The prefixes are excerpts of public corpora and keep the license
of their source: Civil Comments (CC0 1.0); HateXplain, Measuring Hate Speech and Social Bias Frames (CC BY 4.0);
ETHOS (GPL-3.0, 2 prefixes); DynaHate (see its repository). The corpora themselves are not redistributed. The vocabulary
was built from the inventory of [Ferrara et al. (2025)](https://arxiv.org/abs/2507.23319), which is not included.
Details in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#data-sources-and-licenses).

## Citation

```bibtex
@mastersthesis{minini2026lps,
  author = {Minini, Paolo},
  title  = {A probability-based method for measuring post-training effects on lexical preferences in Language Models},
  school = {Universit\`a degli Studi di Milano},
  year   = {2026},
  url    = {https://github.com/paolominini/lexical-preference-shift}
}
```

Supervisor: Prof. Alfio Ferrara. Co-supervisor: Dr. Elisabetta Rocchetti. The larger models were scored on the compute
resources of the host laboratory.

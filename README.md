# Refusal Directions in LLMs — thesis code

Code for studying whether a single "refusal direction" causally mediates refusal in
open-weight instruction-tuned models (Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3,
DeepSeek-R1-Distill-Qwen-7B, with TinyLlama-1.1B-Chat as an early pilot).

## Research questions
- **RQ1** — Are harmful prompts separated from harmless ones in activation space?
- **RQ2** — Does ablating / adding the refusal direction causally change refusal behaviour?
- **RQ3** — Can a Steering-SSR adversarial attack suppress the refusal direction and jailbreak the model?

## Notebooks / scripts
- `01_model_test.ipynb` — pilot: hidden-state extraction, PCA, projection sanity checks (TinyLlama).
- `02_main_experiment.ipynb` — RQ1 (harmfulness separation) and RQ2 (directional ablation +
  activation addition). Defines the 40-prompt dataset, the refusal direction, the ablation/
  addition hooks, and the keyword-based refusal detector.
- `02_main_experiment-Copy1.ipynb` — DeepSeek variant (inner-config hidden-states patch).
- `03_subspace_rerouting.ipynb` — RQ3: Steering-SSR attack (GCG suffix search on a combined
  steering + target-completion objective).
- `resuts-qwen-new/actadd.py`, `results-mistral-new/actadd_mistral.py` — activation-addition runs.
- `results-mistral-new/ablation_mistral.py`, `results-deepseek-new/ablation_deepseek.py` — directional ablation (Mistral, DeepSeek).
- `causal_projection.py` — projects intervention outputs back onto the refusal direction.
- `rq3_summary.py` — builds the RQ3 summary table / figure.

## Reproducing
- **RQ1:** run `02_main_experiment.ipynb` cells 0–6 to extract activations, then the PCA /
  projection cells for per-layer PCA and projection-by-level.
- **RQ2:** ablation = `02_main_experiment.ipynb` cell 8 (or `ablation_mistral.py` /
  `ablation_deepseek.py`); addition = `actadd.py` / `actadd_mistral.py`.
- **RQ3:** `03_subspace_rerouting.ipynb` (steering-only → combined loss → ASCII-constrained
  suffix → easiest-target random restarts).

## Data
- `data/prompts_40.csv` — the 40 prompts (columns: topic, level, prompt); 8 topics × 5 harmfulness levels.
- `data/keyword_lists.txt` — the refusal-detector keyword lists (extended + Arditi short list).

## Supplementary notes
- `docs/deepseek_rq2_note.md` — DeepSeek RQ2 directional-ablation result (refusal-rate table and
  the reasoning-trace / truncation caveats specific to the distilled reasoning model).

## Hardware / environment
Apple M2, 24 GB unified memory, PyTorch MPS backend, models loaded in float16.
Pinned versions in `requirements.txt` (torch 2.11.0, transformers 5.8.0).

## Note on withheld outputs
Generated model responses to harmful prompts, ablated / steered completions, and optimised
adversarial (SSR) suffixes are **deliberately not published** in this repository. The code
reproduces them locally; only aggregate metrics, figures, the prompt set, and the refusal
directions are included.

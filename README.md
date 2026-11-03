# Refusal Mechanisms in LLMs: Experiments (work in progress)

Bachelor's thesis code. Last updated: 3 Nov 2026.

## What this repo does

Studies how language models refuse harmful prompts, by combining behavior (does the model refuse?) with internal states (what do its activations look like?).

- **Dataset:** 40 prompts = 8 topic families x 5 harmfulness levels (0 = harmless, 4 = explicit harmful request).
- **Models:** Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, DeepSeek-R1-Distill-Qwen-7B.
- **Hardware:** Apple M2, 24 GB, PyTorch with MPS.
- **Pipeline:**
  1. Generate a response per prompt and flag refusals with a keyword list.
  2. Extract last-token activations at 7 layers, chosen by relative depth so models are comparable.
  3. PCA plots per layer, colored by harmfulness level.
  4. Refusal direction = mean(Level 4) - mean(Level 0), normalised. Projection scores per prompt.
  5. Steering: subtract the direction at one layer with strength alpha.

Settings (`MODEL_ID`, `MODEL_TAG`) are in the first cell. Result files carry the model tag and `_v2` (e.g. `responses_deepseek_v2.csv`, `deepseek_pca_layer_X.png`).

## Status per research question

| | Status |
|---|---|
| RQ1: is there a level that reliably triggers refusal? | DeepSeek run finished (see caveats). Qwen and Mistral are being rerun today. |
| RQ2: behavior under steering | First attempt (DeepSeek) showed no effect. Switching method, see below. |
| RQ3: prompt modification via subspace rerouting | Not started. |
| Thesis text | Introduction, background and methodology drafted. Results not written yet. |

## Results so far (DeepSeek only, preliminary)

Refusal rate per level (keyword-based, 400 tokens): L0 12.5%, L1 0%, L2 12.5%, L3 37.5%, L4 75%.

Treat these with caution (see issues 3 and 4). Activation results (PCA, projection scores) are unaffected by those issues.

Qwen and Mistral: earlier numbers came from the buggy refusal check (issue 3), so I'm not reporting them. Final numbers will be added after the rerun.

## Problems found, and what was done

1. **DeepSeek returned 1 hidden state instead of 29.** Fixed by setting `model.config.output_hidden_states = True` after loading. Cause not fully understood.
2. **Stuck forward hooks** after a crashed steering run caused confusing errors in later cells. The steering function now removes its hook in a `finally` block.
3. **Refusal-counting bug (affects all three models).** The old `generate_response` returned prompt + answer, so the keyword check also searched the prompt. A prompt containing e.g. "illegally" could count as a refusal. Fixed: only newly generated tokens are checked. Qwen and Mistral are being rerun; I will compare old vs new rates to see how much it changed. PCA and projection results never used this check.
4. **DeepSeek is a reasoning model.** With 400 new tokens, 37 of 40 responses had not finished the thinking phase (no `</think>`), so the keyword check mostly sees reasoning text. Refusal rates for this model are not comparable to the others. Level 0 at 12.5% is a false alarm.
5. **DeepSeek steering had no effect.** Rates were identical for alpha 5, 10, 15, 20, so the push was probably too weak for this model's activation scale. Planned fix below.

## Known limitations

- Keyword-based refusal detection: broad words ("harmful", "illegal") can appear in non-refusals.
- Only 8 prompts per level, so one prompt = 12.5 percentage points.
- Direction is built from Levels 0 and 4 only, so those two separate by construction. The informative part is where Levels 1-3 fall.
- Dataset prompts were written with AI assistance.

## Next steps

1. Finish Qwen and Mistral reruns with the fixed refusal check.
2. Refusal rate per level and per topic, plus manual reading of about 10 responses to check the keyword flags.
3. RQ2: directional ablation as in Arditi et al. (2024), applied at all layers, compared with the baseline per level.
4. Start RQ3.
5. Write results chapters.

## Questions for discussion

- Is keyword-based refusal detection acceptable, or should a sample be checked by hand?
- How much does RQ3 need: one working example, or a systematic evaluation?
- For DeepSeek: disable the thinking phase, or keep it and report the limitation?

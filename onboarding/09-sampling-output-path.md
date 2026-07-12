# 09 — Sampling & Output Path

Primary code:

- [`vllm/v1/sample/sampler.py`](../vllm/v1/sample/sampler.py)
- [`vllm/v1/engine/output_processor.py`](../vllm/v1/engine/output_processor.py)
- [`vllm/outputs.py`](../vllm/outputs.py) — `RequestOutput`

V1 behavior notes: [docs/usage/v1_guide.md](../docs/usage/v1_guide.md) (logprobs modes).

## Where sampling sits

```text
model forward → logits [num_logits_tokens, vocab]
    → Sampler / logits processors
    → sampled token ids
    → Scheduler.update_from_output
    → EngineCoreOutputs
    → OutputProcessor (detokenize, stream)
    → RequestOutput / HTTP chunks
```

Sampling is still on the **worker/GPU side** (or GPU-adjacent), not in the API process.
Detokenization prefers the API/frontend process so EngineCore stays lean.

## Sampler pipeline (authoritative order)

From the `Sampler` class docstring—memorize this order when debugging “wrong”
distributions:

1. Optionally snapshot **raw** logprobs/logits (if requested)
2. Cast logits to float32
3. Allowed-token whitelist
4. Bad-words exclusion
5. Non-argmax-invariant logits processors (min tokens, logit bias, …)
6. Penalties (repetition / frequency / presence)
7. Sample:
   - greedy path when applicable
   - temperature
   - argmax-invariant processors (e.g. min_p)
   - top-k / top-p
   - multinomial (or greedy fallback)
8. Gather top logprobs if requested
9. Return `SamplerOutput`

Penalties before temperature is a common interview/check question—and a common bug
magnet if you reorder ops casually.

## Logprobs semantics (V1 footgun)

By default V1 returns **raw** logprobs (from pre-penalty logits), not the post-penalty
sampling distribution. Modes: `raw_logprobs`, `processed_logprobs`, `raw_logits`,
`processed_logits` via `--logprobs-mode`.

If an eval harness disagrees with HuggingFace-style processed probs, check this before
suspecting the model.

## OutputProcessor responsibilities

- Map token IDs → text (incremental detokenization)
- Handle streaming vs final outputs
- Attach logprobs / finish reasons
- Coordinate request bookkeeping with the frontend

Stop conditions may be detected in the scheduler (EOS, length) *and* involve
output-side string stops—know both exist when debugging “why did it stop?”

## Structured output / grammar (awareness)

Grammar bitmasks can be applied between execute and sample (`get_grammar_bitmask` in
EngineCore). Deep dive is out of scope; know the hook so masks do not surprise you in
profiles.

## Trade-offs

| Topic | Trade-off |
|-------|-----------|
| GPU sampling kernels | Fast batched top-k/p vs complexity |
| Raw vs processed logprobs | Speed / simplicity vs user expectations |
| Streaming detok | Better UX vs more frontend CPU |

## Exercises

1. Trace one greedy request: confirm temperature path is skipped.
2. Request `logprobs=5` and inspect whether values look pre- or post-temperature.
3. Find where `finish_reason` is set in `update_from_output` vs output processor.

Next: [10-performance-profiling.md](10-performance-profiling.md).

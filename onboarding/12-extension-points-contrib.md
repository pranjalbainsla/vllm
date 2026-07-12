# 12 — Extension Points & Contribution Opportunities

Contribution process: [docs/contributing/README.md](../docs/contributing/README.md).

Project AI/accountability rules: [AGENTS.md](../AGENTS.md).

## Safe extension points (after this curriculum)

| Extension | Hook region | Why relatively safe |
|-----------|-------------|---------------------|
| New attention backend | `vllm/v1/attention/backends/` + registry/selector | Models stay unchanged if interface holds |
| Custom logits processors | sampling / logits processor APIs | Localized to sample path |
| Metrics / stats | EngineCore outputs, Prometheus exporters | Observe without changing numerics |
| Scheduler policy experiments | `Scheduler` admit/preempt heuristics | High impact; needs strong tests |
| Triton reference ops | `vllm/v1/attention/ops/` | Good for prototypes |

High-risk without mentorship: changing `BlockPool` hash rules, KV layout, or graph capture
invariants.

## Contribution opportunities aligned with the mental model

These are **example themes**, not assigned tickets—always check open issues/PRs first
(`gh pr list`, issue search) per AGENTS.md.

1. **Scheduler fairness / latency** — reduce decode starvation under heavy prefill;
   measure TTFT/TPOT trade-offs.
2. **KV efficiency** — better utilization metrics, fewer preemptions for common
   patterns; careful APC edge cases.
3. **Runner CPU overhead** — shrink prepare_inputs time visible in Nsight gaps.
4. **Backend capability clarity** — docs/tests for feature matrices when adding flags.
5. **Observability** — expose the metrics you wished you had while debugging chapters
   10–11.

Avoid low-value busywork PRs (typos-only, drive-by renames) unless bundled with
substance—AGENTS.md is explicit.

## Good first investigations (read-only / local)

1. **Annotated step log** — for one offline request, print
   `num_scheduled_tokens` each step until finish; explain the sequence.
2. **Block table audit** — after prefill, assert block count vs `cdiv(len, block_size)`.
3. **Backend identity** — confirm FA is selected; force Triton (if available) and compare
   tokens/s on a tiny model.
4. **Prefix hit demo** — two prompts sharing a long prefix; measure TTFT with APC on/off.
5. **Profiler scavenger hunt** — name the top three CUDA ops in decode; map each to a
   file in this guide.

## How to propose a change (checklist)

1. Duplicate-work search on GitHub issues/PRs
2. Smallest reproduction / benchmark
3. Touch only the layer that owns the bug (scheduler vs runner vs backend)
4. Tests at the cheapest level that catches the failure
5. If model-affecting: include eval notes
6. Human-reviewed; disclose AI assistance per AGENTS.md

## Where *not* to start contributing

Until you have a concrete need:

- Distributed KV transfer / disagg prefill
- Speculative decoding
- Multimodal processors
- Compilation / Inductor deep dives
- New hardware backends

Master the single-GPU continuous-batching loop first—you will move faster later.

## You are ready when…

- You can narrate `EngineCore.step` without notes
- You can point to the file that owns a symptom (queueing vs KV vs attention vs sample)
- You can design a measurement that would falsify your optimization hypothesis

Welcome—and go read the code, not only this guide.

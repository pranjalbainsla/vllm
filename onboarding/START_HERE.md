# Start Here: Fast-Inference Curriculum

**Goal:** Build the mental model required to navigate, profile, optimize, and contribute
to vLLM's V1 engine.

**Time:** ~20–30 hours of focused reading + code walking (not wall-clock calendar time).

**V0 is gone.** Everything that matters lives under `vllm/v1/`. Treat V0 names
(`SequenceGroup`, old `AsyncLLMEngine` internals) as historical vocabulary only.

---

## Before you start

1. Skim [maps/ignore-list.md](maps/ignore-list.md) so you know what *not* to open.
2. Pin [maps/important-files.md](maps/important-files.md)—these ~30 paths are your map.
3. Optional warmup (30–45 min):
   - [docs/usage/v1_guide.md](../docs/usage/v1_guide.md) — why V1 exists
   - [docs/design/arch_overview.md](../docs/design/arch_overview.md) — process layout
   - Blog: [vLLM V1 architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

---

## Reading order (four stages)

### Stage 1 — Concepts + lifecycle (~4–6 h)

| Order | Read | Why |
|------:|------|-----|
| 1 | [00-mental-model.md](00-mental-model.md) | Bottlenecks that justify every later abstraction |
| 2 | [01-request-lifecycle.md](01-request-lifecycle.md) | Types and flow from API → tokens |
| 3 | [traces/offline-generate.md](traces/offline-generate.md) | Concrete offline path |
| 4 | [traces/online-openai.md](traces/online-openai.md) | Concrete online path |
| 5 | [02-engine-architecture.md](02-engine-architecture.md) | Processes, clients, EngineCore |

**Checkpoint — you should be able to:**

- Name the three bottlenecks continuous batching + paging address.
- Trace `LLM.generate` and `vllm serve` to `EngineCore.step` without notes.
- Explain why API tokenization is separated from the GPU-critical loop.

### Stage 2 — Scheduler + KV + memory (~6–8 h) — *core intuition*

| Order | Read | Why |
|------:|------|-----|
| 6 | [03-scheduler-continuous-batching.md](03-scheduler-continuous-batching.md) | Unified token budget |
| 7 | [traces/one-engine-step.md](traces/one-engine-step.md) | One schedule→execute→update cycle |
| 8 | [04-kv-cache-paged-attention.md](04-kv-cache-paged-attention.md) | Why KV is paged |
| 9 | [05-block-manager-memory.md](05-block-manager-memory.md) | BlockPool, layouts, APC |
| 10 | [maps/call-graph.md](maps/call-graph.md) | Cement the one-step call chain |

**Checkpoint — you should be able to:**

- Explain why there is no separate “prefill phase” vs “decode phase” in the V1 scheduler.
- Draw logical block IDs → physical KV slots (`slot_mapping`).
- Predict what happens when KV is exhausted (preemption / recompute).
- Say what Automatic Prefix Caching hashes and why block IDs are append-only.

### Stage 3 — Execute + attention + kernels + sample (~6–8 h)

| Order | Read | Why |
|------:|------|-----|
| 11 | [06-model-execution-pipeline.md](06-model-execution-pipeline.md) | Worker / ModelRunner |
| 12 | [07-attention-backends.md](07-attention-backends.md) | Backend pluggability |
| 13 | [08-kernels-cuda-triton.md](08-kernels-cuda-triton.md) | Layouts + one FA + one Triton path |
| 14 | [09-sampling-output-path.md](09-sampling-output-path.md) | Logits → tokens → text |
| 15 | [maps/dependency-map.md](maps/dependency-map.md) | Module relationships |

**Checkpoint — you should be able to:**

- Walk `GPUModelRunner.execute_model` at the level of `_update_states` → forward → sample.
- Point to where attention metadata is built and how FlashAttn is selected.
- Describe what `block_tables` and `slot_mapping` mean to a kernel.
- List the Sampler pipeline steps in order.

### Stage 4 — Profile, debug, contribute (~4–6 h)

| Order | Read | Why |
|------:|------|-----|
| 16 | [10-performance-profiling.md](10-performance-profiling.md) | What to measure and how |
| 17 | [11-debugging-pitfalls.md](11-debugging-pitfalls.md) | Failure modes and tips |
| 18 | [12-extension-points-contrib.md](12-extension-points-contrib.md) | Where to change code safely |

**Checkpoint — you should be able to:**

- Name TTFT / TPOT / KV util / preemption rate and where each appears.
- Avoid the top pitfalls (V0 docs, OOM from graphs+KV, block_id vs offset).
- Propose a non-trivial contribution that touches the hot path with a test plan.

---

## Recommended code-walking order (alongside chapters)

Do not read entire files top-to-bottom. Jump to these symbols:

1. `EngineCore.step` — `vllm/v1/engine/core.py`
2. `Scheduler.schedule` / `update_from_output` — `vllm/v1/core/sched/scheduler.py`
3. `KVCacheManager` allocate/free — `vllm/v1/core/kv_cache_manager.py`
4. `GPUModelRunner.execute_model` — `vllm/v1/worker/gpu_model_runner.py`
5. `FlashAttentionImpl.forward` — `vllm/v1/attention/backends/flash_attn.py`
6. `Sampler.forward` — `vllm/v1/sample/sampler.py`

---

## Official docs to keep nearby

| Doc | Use when |
|-----|----------|
| [docs/design/arch_overview.md](../docs/design/arch_overview.md) | Process / class hierarchy |
| [docs/usage/v1_guide.md](../docs/usage/v1_guide.md) | V1 behavior differences |
| [docs/design/prefix_caching.md](../docs/design/prefix_caching.md) | APC hashing details |
| [docs/design/attention_backends.md](../docs/design/attention_backends.md) | Backend feature matrix |
| [docs/design/cuda_graphs.md](../docs/design/cuda_graphs.md) | Graph capture modes |
| [docs/contributing/profiling.md](../docs/contributing/profiling.md) | How to profile |
| [docs/design/paged_attention.md](../docs/design/paged_attention.md) | **Historical only** — not current code |

---

## After the curriculum

- Re-read [maps/important-files.md](maps/important-files.md) and mark which files you can
  explain from memory.
- Pick one exercise from [12-extension-points-contrib.md](12-extension-points-contrib.md).
- Only then open ignored areas (distributed, multimodal, spec decode) with a clear goal.

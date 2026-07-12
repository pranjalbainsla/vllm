# 06 — Model Execution Pipeline

Primary code:

- [`vllm/v1/executor/abstract.py`](../vllm/v1/executor/abstract.py) — executor factory
- [`vllm/v1/executor/multiproc_executor.py`](../vllm/v1/executor/multiproc_executor.py)
- [`vllm/v1/worker/gpu_worker.py`](../vllm/v1/worker/gpu_worker.py)
- [`vllm/v1/worker/gpu_model_runner.py`](../vllm/v1/worker/gpu_model_runner.py)
- [`vllm/forward_context.py`](../vllm/forward_context.py)

Skip for now: Model Runner V2 (`vllm/v1/worker/gpu/`, `docs/design/model_runner_v2.md`).

## Why this layer exists

The scheduler speaks in request IDs and block IDs. The GPU speaks in tensors and kernels.
The **model runner** is the translation layer: persistent batch state, input buffers,
attention metadata, cudagraph dispatch, and sampling.

Without it, every model file would re-implement padding, paged KV indexing, and
batching—an unmaintainable mess.

## Call chain

```text
EngineCore.step
  └─ Executor.execute_model(SchedulerOutput)
       └─ GPUWorker.execute_model
            └─ GPUModelRunner.execute_model
                 ├─ _update_states      # persistent batch ↔ scheduler diff
                 ├─ _prepare_inputs     # tokens, positions, slot_mapping, …
                 ├─ (cudagraph) model forward
                 └─ sample_tokens      # may be deferred to a second call
```

## Persistent batch (intuition)

The runner keeps GPU-side buffers sized for `max_num_seqs` / `max_num_batched_tokens`.
Each step:

1. **`_update_states`** — add new requests, remove finished, append block IDs for cached
   ones, sync sampling metadata
2. **`_prepare_inputs`** — pack this step's tokens into contiguous input tensors; build
   attention metadata into `ForwardContext`
3. **Forward** — `nn.Module` runs; each `Attention` layer reads metadata from context
4. **Sample** — logits → token IDs (see chapter 09)

This avoids allocating brand-new Python structures for the entire batch from scratch
when only one request changed.

## `ForwardContext`

Attention backends need per-step metadata (seqlens, block tables, …) without threading
hundreds of args through every model layer. `ForwardContext` / `get_forward_context()`
is the ambient channel. When debugging “wrong attention,” check whether metadata was
built for *this* step before forward.

## Async execute / sample split

`EngineCore.step` may see `execute_model` return `None`, then call `sample_tokens`. That
enables overlapping: GPU compute finishes while CPU prepares grammar bitmasks / next
schedule work. State machine warning in runner:

> `sample_tokens()` must be called after `execute_model()` returns `None`.

If you add code between them, you can corrupt `execute_model_state`.

## CUDA graphs (conceptual)

Decode (and some mixed shapes) pay a tax on many small kernel launches. CUDA graphs
capture a replayable launch sequence for fixed shapes/modes.

Trade-offs:

- **+** Lower CPU launch overhead, better TPOT at scale
- **−** Extra memory at capture; shape constraints; harder debugging (graph vs eager)

Read [docs/design/cuda_graphs.md](../docs/design/cuda_graphs.md) when optimizing—not
before you can follow an eager forward.

## Executor variants

| Kind | When |
|------|------|
| Uni / in-process worker | Simple / single GPU paths |
| Multiproc executor | Default multi-GPU TP workers |
| Ray / external launcher | Distributed clusters — **ignore initially** |

TP=1 still uses the worker/runner abstractions; do not skip this chapter on single GPU.

## Bottleneck focus here

When profiles show:

| Symptom | Look at |
|---------|---------|
| Long gaps before GPU kernels | `_prepare_inputs`, scheduling metadata, D2H/H2D |
| Low GPU util during decode | batch size, graphs disabled, tiny `max_num_seqs` |
| Spikes on new requests | state update + graph warmup / capture |

## Exercises

1. Skim `execute_model` from the start until the model forward call; list the major
   `with record_function...` regions.
2. Find where block IDs from `SchedulerOutput` enter `BlockTable`.
3. Note one place sampling is deferred vs done inline.

Next: [07-attention-backends.md](07-attention-backends.md).

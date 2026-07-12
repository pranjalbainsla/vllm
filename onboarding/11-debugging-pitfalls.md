# 11 — Debugging Tips & Common Pitfalls

## Pitfalls (read twice)

### 1. Treating V0 docs / blogs as current

V0 engine codepaths are gone. `SequenceGroup`, old scheduler phases, and many blog
diagrams are historical. Prefer `vllm/v1/` and
[docs/usage/v1_guide.md](../docs/usage/v1_guide.md).

Especially: [docs/design/paged_attention.md](../docs/design/paged_attention.md) is
**explicitly outdated** relative to today's FlashAttention-based stack.

### 2. Confusing `block_id` with sequence position or byte offset

- `block_id` — pool index / table entry
- sequence position — token index
- slot — `block_id * block_size + offset` (typical)

Wrong mental model → wrong kernel debugging.

### 3. OOM from graphs + KV + weights

Raising `gpu_memory_utilization` is not free. Capture memory for CUDA graphs competes
with the KV pool. Symptom: works with eager, OOMs with graphs (or vice versa at
startup).

### 4. Preemption storms misdiagnosed as “scheduler bugs”

If logs show frequent preemption, you likely oversubscribed KV (context length ×
concurrency × layers). Fix memory/concurrency first.

### 5. APC hash mismatches / unexpected misses

Prefix caching only hits on **full blocks** with matching hash extras (LoRA, multimodal
inputs, cache salt, …). Partial-block prefixes do not hit. “APC does nothing” is often
workload shape, not a broken pool.

### 6. Logprobs disagreement with HF

V1 default raw logprobs ≠ post-penalty probs. Check `--logprobs-mode` before diving into
Sampler bugs.

### 7. Async scheduling state machine

If `execute_model` returned `None`, you **must** call `sample_tokens` before the next
execute. Custom hooks that break this will throw or corrupt state.

### 8. Debugging only the GPU while the API process is stuck

Online hangs: verify ZMQ client/server, EngineCore busy loop, and waiting queue—not only
`nvidia-smi`.

## Debugging playbook

### Request never produces tokens

1. Did `InputProcessor` emit an `EngineCoreRequest`?
2. Is the request on `waiting` or `running`?
3. Is `num_scheduled_tokens` ever non-zero for it?
4. Are outputs reaching `OutputProcessor`?

### Wrong outputs / garbage text

1. Confirm tokenizer/model pair and chat template (frontend).
2. Check stop/EOS handling.
3. Verify KV: prefix cache disabled still wrong? → model/sampling; only with APC → hash
   / sharing suspicion.
4. Backend: force eager, known-good FA version.

### Performance cliff

1. Measure TTFT vs TPOT separately.
2. Check running/waiting counts and KV free blocks over time.
3. Profile one steady decode phase after warmup.
4. Only then inspect attention kernels.

## Practical tools

| Tool | Tip |
|------|-----|
| `VLLM_LOGGING_LEVEL=DEBUG` | Verbose; narrow to module loggers when possible |
| Breakpoints in `EngineCore.step` | Best under inproc offline `LLM` |
| Dump `SchedulerOutput` | `num_scheduled_tokens`, new vs cached reqs |
| `dump_engine_exception` helpers | Used on engine failures—read those dumps |
| `anon_repr` on request data | Safer logging without prompt leakage |

## Exercises

1. Intentionally set tiny KV (low `gpu_memory_utilization`) and trigger preemption; read
   the log line and map it to scheduler code.
2. Disable prefix caching and compare TTFT on a shared-prefix workload.
3. Break async sample ordering in a local hack and observe the error—then revert.

Next: [12-extension-points-contrib.md](12-extension-points-contrib.md).

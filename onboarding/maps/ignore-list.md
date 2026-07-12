# Safe to Ignore Initially

Open these only with a concrete goal. They are important to vLLM as a product, but they
distract from the single-GPU continuous-batching mental model.

| Area | Paths | Why skip at first |
|------|-------|-------------------|
| Distributed / DP / PP | `vllm/distributed/`, `vllm/v1/engine/coordinator.py` | Multi-engine orchestration |
| Ray executors | `vllm/v1/executor/ray_*.py` | Cluster launch complexity |
| Multimodal | `vllm/multimodal/`, encoder-cache branches in scheduler/runner | Extra encoder pipeline |
| Speculative decoding | `vllm/v1/spec_decode/`, `vllm/v1/worker/gpu/spec_decode/` | Draft/verify on same scheduler |
| KV transfer / disagg / offload | `vllm/distributed/kv_transfer/`, `vllm/v1/kv_offload/` | P–D disaggregation |
| Plugins | `vllm/plugins/` | Optional loaders |
| LoRA | `vllm/lora/` | Adapter overlay |
| Structured output | `vllm/v1/structured_output/` | Grammar machinery |
| Exotic attention backends | MLA, Mamba, ROCm, XPU, TPU, CPU, Flex, TurboQuant under `vllm/v1/attention/backends/` | Start with `FLASH_ATTN` (+ Triton for reading) |
| Model Runner V2 | `vllm/v1/worker/gpu/`, `docs/design/model_runner_v2.md` | Parallel redesign; default remains `gpu_model_runner.py` |
| Compilation deep dive | `vllm/compilation/`, most of `docs/design/torch_compile.md` | After you know eager forward |
| Pooling / embeddings | `vllm/v1/pool/`, pooling entrypoints | Different task |
| Historical paged kernel doc | `docs/design/paged_attention.md` | Does not match current FA codepaths |
| Rust frontend | `rust/` | Separate surface |

## “Ignore” does not mean “unimportant”

When your team owns disagg serving or a TPU backend, return here with the core loop
already internalized—you will recognize which pieces are reused (scheduler token budget,
block tables, EngineCore step) versus which are new.

## How to know you are ready to open an ignored area

- You can explain `EngineCore.step` and `SchedulerOutput` without notes
- You can point to where KV blocks become `slot_mapping`
- You have profiled at least one decode-heavy run

Then pick **one** ignored area and map it onto the core loop deliberately.

# ~30 Most Important Files & Directories

Master these before wandering the rest of the tree. Each entry: **read for** / **skip**.

## Entry & request surface

| Path | Read for | Skip initially |
|------|----------|----------------|
| [`vllm/entrypoints/llm.py`](../vllm/entrypoints/llm.py) | Offline `LLM.generate` entry | Fancy offline extras |
| [`vllm/entrypoints/openai/api_server.py`](../vllm/entrypoints/openai/api_server.py) | How HTTP reaches AsyncLLM | Every route / middleware |
| [`vllm/v1/engine/async_llm.py`](../vllm/v1/engine/async_llm.py) | Online frontend loop | Edge-case pause/RPC |
| [`vllm/v1/engine/llm_engine.py`](../vllm/v1/engine/llm_engine.py) | Sync frontend / offline | Legacy naming history |
| [`vllm/v1/engine/core_client.py`](../vllm/v1/engine/core_client.py) | Inproc vs MP clients | Every ZMQ variant detail |
| [`vllm/v1/engine/core.py`](../vllm/v1/engine/core.py) | `EngineCore.step`, KV init | DP/elastic/EP branches |
| [`vllm/v1/engine/input_processor.py`](../vllm/v1/engine/input_processor.py) | Prompt → `EngineCoreRequest` | Multimodal branches |
| [`vllm/v1/engine/output_processor.py`](../vllm/v1/engine/output_processor.py) | Tokens → `RequestOutput` | Pooling outputs |
| [`vllm/v1/request.py`](../vllm/v1/request.py) | Scheduler `Request` state | Speculative fields deep dive |
| [`vllm/v1/engine/__init__.py`](../vllm/v1/engine/__init__.py) | Wire types | Rare notification enums |

## Scheduler & KV

| Path | Read for | Skip initially |
|------|----------|----------------|
| [`vllm/v1/core/sched/scheduler.py`](../vllm/v1/core/sched/scheduler.py) | Continuous batching | Encoder/spec/DP throttle corners |
| [`vllm/v1/core/sched/output.py`](../vllm/v1/core/sched/output.py) | `SchedulerOutput` contract | Connector metadata |
| [`vllm/v1/core/sched/interface.py`](../vllm/v1/core/sched/interface.py) | `SchedulerInterface` | Pause enum minutiae |
| [`vllm/v1/core/kv_cache_manager.py`](../vllm/v1/core/kv_cache_manager.py) | Allocate/free facade | Hybrid edge cases |
| [`vllm/v1/core/block_pool.py`](../vllm/v1/core/block_pool.py) | Free list + APC map | Event emission details |
| [`vllm/v1/core/kv_cache_utils.py`](../vllm/v1/core/kv_cache_utils.py) | `KVCacheBlock`, hashing helpers | Every hash helper |
| [`vllm/v1/core/kv_cache_coordinator.py`](../vllm/v1/core/kv_cache_coordinator.py) | Multi-group awareness | Full hybrid algorithms |
| [`vllm/v1/kv_cache_interface.py`](../vllm/v1/kv_cache_interface.py) | `KVCacheSpec` / config | Every spec subclass |

## Execution

| Path | Read for | Skip initially |
|------|----------|----------------|
| [`vllm/v1/executor/abstract.py`](../vllm/v1/executor/abstract.py) | Executor factory | Ray/external launchers |
| [`vllm/v1/executor/multiproc_executor.py`](../vllm/v1/executor/multiproc_executor.py) | Default worker orchestration | Failure recovery deep dive |
| [`vllm/v1/worker/gpu_worker.py`](../vllm/v1/worker/gpu_worker.py) | One GPU process | Device sleep / profiler hooks |
| [`vllm/v1/worker/gpu_model_runner.py`](../vllm/v1/worker/gpu_model_runner.py) | `execute_model` / prepare | Spec decode, MM, MRV2 |
| [`vllm/v1/worker/block_table.py`](../vllm/v1/worker/block_table.py) | `slot_mapping` | CP interleave modes |
| [`vllm/forward_context.py`](../vllm/forward_context.py) | Attn metadata channel | Every cudagraph enum |

## Attention, sampling, kernels

| Path | Read for | Skip initially |
|------|----------|----------------|
| [`vllm/model_executor/layers/attention/attention.py`](../vllm/model_executor/layers/attention/attention.py) | `Attention` → backend | Quant/KV-sharing branches |
| [`vllm/v1/attention/selector.py`](../vllm/v1/attention/selector.py) | Backend selection | Every flag combination |
| [`vllm/v1/attention/backends/registry.py`](../vllm/v1/attention/backends/registry.py) | Enum of backends | Non-FA entries |
| [`vllm/v1/attention/backends/flash_attn.py`](../vllm/v1/attention/backends/flash_attn.py) | Production FA path | MLA/DCP |
| [`vllm/v1/attention/ops/`](../vllm/v1/attention/ops/) | Readable Triton ops | TurboQuant / exotic |
| [`vllm/v1/sample/sampler.py`](../vllm/v1/sample/sampler.py) | Logits → token IDs | Every processor variant |
| [`csrc/`](../csrc/) + [`vllm/_custom_ops.py`](../vllm/_custom_ops.py) | Where CUDA binds in | All op implementations |

## Config

| Path | Read for | Skip initially |
|------|----------|----------------|
| [`vllm/config/`](../vllm/config/) | `VllmConfig`, scheduler/cache/compilation | Every nested dataclass |

## Official design docs (pair with code)

| Path | Read for |
|------|----------|
| [`docs/design/arch_overview.md`](../docs/design/arch_overview.md) | Processes & hierarchy |
| [`docs/usage/v1_guide.md`](../docs/usage/v1_guide.md) | V1 behavior |
| [`docs/design/prefix_caching.md`](../docs/design/prefix_caching.md) | APC |
| [`docs/design/attention_backends.md`](../docs/design/attention_backends.md) | Backend matrix |
| [`docs/contributing/profiling.md`](../docs/contributing/profiling.md) | How to profile |

**Count:** 30+ code paths in the tables above if you include `ops/` and `config/` as directories—that is intentional. Do not try to memorize line numbers; memorize *ownership*.

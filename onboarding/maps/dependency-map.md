# Module Dependency Map

High-level dependencies for the fast-inference core. Arrows mean “uses / calls.”

```mermaid
flowchart TB
  subgraph entry [Entrypoints]
    LLM[entrypoints_llm]
    API[entrypoints_openai_api_server]
  end

  subgraph fe [Frontend_V1]
    AsyncLLM[v1_engine_async_llm]
    LLMEngine[v1_engine_llm_engine]
    InProc[input_processor]
    OutProc[output_processor]
    Client[engine_core_client]
  end

  subgraph core [EngineCore]
    EC[v1_engine_core]
    Sched[v1_core_sched_scheduler]
    KVM[v1_core_kv_cache_manager]
    Pool[v1_core_block_pool]
  end

  subgraph exec [Execution]
    Exe[v1_executor]
    Worker[v1_worker_gpu_worker]
    Runner[v1_worker_gpu_model_runner]
    BT[v1_worker_block_table]
  end

  subgraph model [Model_Attention]
    Mod[model_executor_models]
    Attn[layers_attention]
    Sel[v1_attention_selector]
    FA[v1_attention_backends_flash_attn]
    Ops[v1_attention_ops_and_csrc]
    Samp[v1_sample_sampler]
  end

  LLM --> LLMEngine
  API --> AsyncLLM
  LLMEngine --> InProc
  AsyncLLM --> InProc
  LLMEngine --> OutProc
  AsyncLLM --> OutProc
  LLMEngine --> Client
  AsyncLLM --> Client
  Client --> EC
  EC --> Sched
  EC --> Exe
  Sched --> KVM
  KVM --> Pool
  Exe --> Worker
  Worker --> Runner
  Runner --> BT
  Runner --> Mod
  Mod --> Attn
  Attn --> Sel
  Sel --> FA
  FA --> Ops
  Runner --> Samp
  Sched -->|"SchedulerOutput"| Runner
  Samp -->|"ModelRunnerOutput"| Sched
```

## Notes

- Config (`vllm/config`) fans into almost every node—omitted to reduce clutter.
- Multimodal, spec decode, and distributed KV would hang off scheduler/runner/executor;
  deliberately omitted ([ignore-list.md](ignore-list.md)).

See also [call-graph.md](call-graph.md) for the per-step call sequence.

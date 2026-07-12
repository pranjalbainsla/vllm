# Trace: Offline `LLM.generate`

**Goal:** Follow one offline completion from Python API into `EngineCore` and back.

## Entry

```python
from vllm import LLM, SamplingParams
llm = LLM(model="...")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=16))
```

Primary file: [`vllm/entrypoints/llm.py`](../vllm/entrypoints/llm.py).

`LLM` constructs an `LLMEngine` ([`vllm/v1/engine/llm_engine.py`](../vllm/v1/engine/llm_engine.py)),
which owns:

- `InputProcessor` — prompt → `EngineCoreRequest`
- `OutputProcessor` — `EngineCoreOutput` → `RequestOutput`
- `EngineCoreClient` — usually in-process for offline

## Call chain (simplified)

```text
LLM.generate
  └─ LLM._run_completion / engine methods
       └─ LLMEngine.add_request
            ├─ InputProcessor.process_inputs  → EngineCoreRequest
            └─ engine_core.add_request        → Request on scheduler.waiting
       └─ loop: LLMEngine.step / get outputs
            └─ EngineCore.step
                 ├─ Scheduler.schedule
                 ├─ Executor.execute_model → GPUModelRunner
                 ├─ (optional) sample_tokens
                 └─ Scheduler.update_from_output → EngineCoreOutputs
            └─ OutputProcessor.process_outputs → RequestOutput
```

## What to watch in a debugger

| Breakpoint | Inspect |
|------------|---------|
| `InputProcessor` output | `request_id`, `prompt_token_ids`, `sampling_params` |
| `Scheduler.schedule` return | `num_scheduled_tokens`, `scheduled_new_reqs` |
| `GPUModelRunner.execute_model` | batch size, positions, block tables |
| `Sampler` / sample path | sampled token ids |
| `OutputProcessor` | detokenized text, `finished` flag |

## First step vs later steps

- **First step(s):** request often appears in `NewRequestData` with full prompt tokens and
  initial block IDs; may schedule many prefill tokens (or a chunk).
- **Later steps:** request is in `CachedRequestData`; typically `num_scheduled_tokens=1`
  per decode step (more if speculative decoding—ignore for now).

## Differences from online

| | Offline `LLM` | Online `AsyncLLM` |
|--|---------------|-------------------|
| Driving loop | Caller polls / runs until done | Background output handler + EngineCoreProc |
| Client | Often inproc | Often async MP + ZMQ |
| Streaming | Usually final `RequestOutput` | Incremental async generator |

Continue with [online-openai.md](online-openai.md) and [one-engine-step.md](one-engine-step.md).

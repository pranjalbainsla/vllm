# Diagram: GPU memory hierarchy at init

```mermaid
flowchart TB
  GPU[GPU_HBM]
  GPU --> Weights[Model_weights]
  GPU --> Acts[Activation_workspace]
  GPU --> Graphs[CUDA_graph_memory]
  GPU --> KVPool[KV_cache_block_pool]
  GPU --> Other[Allocator_fragmentation_and_runtime]

  KVPool --> Blocks[Fixed_size_blocks]
  Blocks --> LayerTensors[Per_layer_or_shared_KVCacheTensor]
```

**Init order that matters:**

1. Load / shard weights
2. Profile peak activation (+ graph capture estimates)
3. Give the remainder (scaled by `gpu_memory_utilization`) to the KV block pool
4. Scheduler concurrency is then bounded by `num_gpu_blocks`

Use with: [05-block-manager-memory.md](../05-block-manager-memory.md),
[02-engine-architecture.md](../02-engine-architecture.md).

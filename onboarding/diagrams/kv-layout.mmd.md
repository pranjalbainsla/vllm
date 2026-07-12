# Diagram: logical blocks → physical KV

```mermaid
flowchart LR
  subgraph logical [Per_request_logical_view]
    T0[Tokens_0_15]
    T1[Tokens_16_31]
    T2[Tokens_32_47]
    BT[block_table_ids]
    T0 --> BT
    T1 --> BT
    T2 --> BT
  end

  subgraph physical [GPU_KV_tensors]
    B7[Physical_block_7]
    B2[Physical_block_2]
    B19[Physical_block_19]
  end

  BT -->|"id_7"| B7
  BT -->|"id_2"| B2
  BT -->|"id_19"| B19

  SM[slot_mapping_token_to_slot] --> B7
  SM --> B2
  SM --> B19
```

**Invariant:** block IDs in the table are *logical handles* into a pool; they need not be
contiguous. Attention kernels index K/V through `block_tables` + `slot_mapping`, not
through a contiguous `[seq, pos]` tensor.

Use with: [04-kv-cache-paged-attention.md](../04-kv-cache-paged-attention.md),
[05-block-manager-memory.md](../05-block-manager-memory.md).

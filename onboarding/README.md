# vLLM Fast-Inference Onboarding

This folder is a **study guide for engineers who want to understand and contribute to
vLLM's core inference engine**—not a user manual for calling the library.

It teaches the systems concepts behind fast LLM serving (continuous batching, paged KV
cache, scheduling, attention backends) and maps them onto the **V1** implementation
under `vllm/v1/`.

## How to use this

1. Open **[START_HERE.md](START_HERE.md)** and follow the staged reading order.
2. Keep [maps/important-files.md](maps/important-files.md) and
   [maps/ignore-list.md](maps/ignore-list.md) open as you read.
3. When a chapter links into `docs/design/` or source files, follow those links—they
   are the source of truth. This guide sequences and explains; official docs win on
   details.

## What this is not

- Not part of the published MkDocs site under `docs/`.
- Not a replacement for [docs/design/arch_overview.md](../docs/design/arch_overview.md)
  or [docs/usage/v1_guide.md](../docs/usage/v1_guide.md).
- Not coverage of distributed serving, multimodal, speculative decoding, LoRA, or
  exotic hardware—see the ignore list.

## Prerequisites

- Comfortable with Transformer attention and autoregressive decoding
- Basic CUDA / GPU memory model vocabulary (HBM, kernels, occupancy)—not kernel authoring
- Python concurrency awareness (`asyncio`, multiprocessing) helps for the engine chapter

## Structure

| Path | Purpose |
|------|---------|
| `00`–`12-*.md` | Curriculum chapters |
| `maps/` | File map, ignore list, dependency and call graphs |
| `traces/` | Annotated request and engine-step walkthroughs |
| `diagrams/` | Standalone Mermaid sources reused by chapters |

## Mentoring stance

Read as if a maintainer is walking you through the codebase: every abstraction should
answer **what bottleneck it solves**, **what trade-offs it accepts**, and **where to
look in the tree**. When you finish, you should be able to navigate a profile,
hypothesize a bottleneck, and open a meaningful PR—not just run `LLM.generate`.

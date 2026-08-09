# ADR-001: Language and runtime

## Status
Accepted

## Context
Need core language for controller + GPU agent + scheduler. Must support gRPC well, concurrency-safe state, race detection tooling, and integrate with CUDA/PyTorch/NCCL tooling for real-GPU validation.

## Decision
Go for controller, GPU agent, scheduler, APIs, metrics. Python used only at the edges: CUDA/PyTorch/NCCL-backed real-hardware probing scripts and workload simulation harnesses where PyTorch ecosystem tooling is the natural fit. Go never shells out to Python for anything on the scheduling hot path.

## Tradeoffs
Go gives strong concurrency primitives (goroutines, channels, context), a built-in race detector, and static binaries good for Docker Compose sim fleets. Cost: no first-class CUDA bindings — real-GPU capability probing goes through NVML via cgo bindings or shelling to `nvidia-smi`/a thin Python probe process, not native Go CUDA calls. Accepted because scheduling logic (the actual point of this project) never needs to touch CUDA directly — it only needs GPU *metadata*.

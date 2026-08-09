# Benchmark plan

Hard rule: no number in any doc/README is estimated. Every reported number traces to a file under `benchmark/results/`.

## Metrics to measure

1. **Scheduling latency (P50/P95/P99)** — wall-clock time from workload entering QUEUED to a PlacementResult being produced, measured per policy, across a range of fleet sizes (8/16 simulated workers) and load levels.
2. **Queue latency** — time from SUBMITTED to SCHEDULED (includes time waiting for admission + waiting in queue for a free GPU).
3. **Allocation success rate** — fraction of submitted workloads that reach RUNNING vs. FAILED/CANCELLED due to unsatisfiable requirements, under a fixed synthetic workload mix.
4. **GPU utilization** — average and distribution of per-GPU utilization over a benchmark run (simulated utilization model must be documented and deterministic).
5. **Fragmentation** — measure of unusable "stranded" GPU memory/count across the fleet after a run (e.g. GPUs with free memory too small for any queued workload).
6. **Scheduling throughput** — placements/sec the scheduler can produce under load (isolates scheduler-only cost from queueing effects).
7. **Topology-aware improvement** — head-to-head comparison: fraction of NVLink-required workloads placed correctly by Topology Aware vs. First Fit/Best Fit, and any latency cost of the topology-aware search.

## Methodology

- All benchmarks run against the simulator with a fixed, recorded seed — reproducibility is mandatory; seed goes into the result file.
- Each policy benchmarked under identical synthetic workload traces (same submission times, same requirement mix) for fair comparison.
- Every run's raw output written as JSON to `benchmark/results/<benchmark-name>-<timestamp>.json`, containing: seed, fleet config, workload trace summary, policy name, raw latency samples (or histogram), and derived percentiles.
- `benchmark/results/` is machine-written only — never hand-edited. README/docs pull numbers directly from these files.
- Failure-recovery-adjacent timing (e.g. reschedule latency after a simulated worker failure) measured the same way: real timestamps from a real simulated run, not estimated.

## What "done" looks like for a benchmark

A given benchmark is considered valid only when: it ran against the actual code (not a mock), used a fixed documented seed, wrote a result file, and that result file is what any README table cites — with a path reference, not just a number.

## Real-hardware benchmarks

When real CUDA hardware is available, the same benchmark harness runs against real GPU agents. Results are written to a separate `benchmark/results/real/` subdirectory and explicitly labeled `hardware_mode: real` in the result JSON so simulated and real numbers are never conflated in the README.

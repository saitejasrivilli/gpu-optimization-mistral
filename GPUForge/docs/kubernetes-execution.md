# Kubernetes execution backend (Phase 5)

Package: `internal/k8sexec`. The *only* GPUForge package that imports `k8s.io/client-go` (or any Kubernetes package). `internal/domain`, `internal/scheduler`, and `internal/orchestrator` remain entirely Kubernetes-independent — this was verified by grep, not just asserted:

```
grep -rl 'k8s.io' internal/domain internal/scheduler internal/orchestrator   # -> no matches
```

## Architecture

```
Orchestrator (internal/orchestrator, unchanged from Phase 4)
    |
    | orchestrator.Executor interface  (Start / Status / Cancel — unchanged)
    |
    +----------------------+----------------------------+
    |                                                   |
SimulatedExecutor (Phase 4)                  KubernetesExecutor (Phase 5)
    |                                                   |
in-memory, deterministic                     one batchv1.Job per workload
                                                         |
                                                    Kubernetes API
                                                         |
                                              kubelet + NVIDIA device plugin
                                                         |
                                                 real GPU hardware (if present)
```

`KubernetesExecutor` implements `orchestrator.Executor` with no new methods and no interface changes — the orchestrator code that calls `Start`/`Status`/`Cancel` is identical regardless of which executor it holds. GPUForge is not, and does not become, a general Kubernetes operator: it creates exactly one resource kind (`batchv1.Job`), for exactly one purpose (running a workload GPUForge already decided how to place), and touches nothing else in the cluster.

## KubernetesExecutor

```go
func New(client kubernetes.Interface, namespace, image string, logger *slog.Logger) *KubernetesExecutor
```

`client` is `kubernetes.Interface` — a real clientset (built from a kubeconfig, see Local setup) in production, or `k8s.io/client-go/kubernetes/fake` in unit tests (`internal/k8sexec/executor_test.go`), so the executor's own logic is fully testable without a cluster.

- **Start**: checks for an existing Job by deterministic name first (`Get`, not a local cache — see Reconciliation safety), then creates one. Returns `orchestrator.ErrAlreadyStarted` if a Job already exists, whether from an earlier call in this process or discovered fresh after a restart.
- **Status**: `Get`s the Job and maps its live state to an `orchestrator.ExecutionStatus` — see Status reconciliation below. Never assumes `Create == Running`.
- **Cancel**: idempotent Job deletion — see Cancellation below.

## Workload model / GPU resource mapping

Every Job GPUForge creates:

| GPUForge concept | Kubernetes representation |
|---|---|
| Workload identity | Deterministic Job name (`jobNameFor`) + `gpuforge.io/workload-id` annotation carrying the exact, unsanitized ID |
| GPU count | `resources.limits["nvidia.com/gpu"]` on the single container |
| Container image | `KubernetesExecutor.image`, fixed per executor instance (one image per deployment of GPUForge, not per-workload — workload *type* differentiation is a later-phase concern) |
| Environment/configuration | `GPUFORGE_WORKLOAD_ID`, `GPUFORGE_WORKER_ID`, `GPUFORGE_GPU_IDS` env vars |
| Restart behavior | `RestartPolicy: Never`, `BackoffLimit: 0` |
| Labels for reconciliation | `app=gpuforge`, `component=workload-executor`, `app.kubernetes.io/managed-by=gpuforge` |

**Fields deliberately not modeled**: CPU/memory requests (the domain's `WorkloadRequirements` doesn't carry them — see docs/scheduling-engine.md; nothing to map), custom container command/args (every workload runs the same image's default entrypoint — see cmd/workload's placeholder; a future phase that supports heterogeneous workload images/commands would extend `ExecutionRequest`, not invent ad hoc fields here), node affinity beyond the opt-in `NodeSelectorKey` (see below).

### Why `BackoffLimit: 0` / `RestartPolicy: Never`

Retry policy is GPUForge's own (`internal/orchestrator.RetryPolicy`), already implemented in Phase 4. Letting the Kubernetes Job controller *also* retry would mean two independent, uncoordinated retry policies fighting over the same failure — worse than either alone. GPUForge always sees the first failure and decides.

### GPUForge's placement vs. Kubernetes' execution — an explicit distinction

**GPUForge's scheduler (`internal/scheduler`) already chose which worker and which specific GPU IDs a workload should use, before `KubernetesExecutor.Start` is ever called.** Kubernetes does not re-run that decision and cannot be told "use exactly GPU UUID X" through the `nvidia.com/gpu` extended resource — it's a *count*, not a device selector. The NVIDIA device plugin on the target node picks the actual physical device(s) at kubelet level, independently of GPUForge's own reasoning about which GPU (by ID) was "best." In a deployment where every GPUForge worker maps 1:1 to a Kubernetes node with exactly the GPUs GPUForge's registry believes it has, this discrepancy is invisible in practice; in any other topology, it is a real gap — Kubernetes might schedule the pod's requested GPU count onto *a* GPU on the target node, not necessarily the *specific* GPU GPUForge's `TopologyAware` policy reasoned about. This is documented here rather than hidden, per Phase 0's "never claim a number/behavior you didn't verify" principle.

`NodeSelectorKey` (opt-in, empty by default) can pin a Job's pod to a specific node via `nodeSelector[key] = req.WorkerID`, for clusters where GPUForge worker IDs are known to equal real node names — this narrows but does not eliminate the device-selection gap above (it still doesn't pick the specific GPU device).

## Status reconciliation

Explicit `batchv1.JobStatus -> orchestrator.ExecutionPhase` mapping (`jobToExecutionStatus`):

| Kubernetes state | GPUForge `ExecutionPhase` |
|---|---|
| Job just created, no status yet | `Running` |
| Pod Pending | `Running` |
| Pod Running | `Running` |
| `Status.Succeeded >= 1` | `Succeeded` |
| `Status.Failed >= 1` | `Failed` (reason from the `JobFailed` condition when present; `Retryable: true` always — see Limitations) |
| Job not found (`Get` returns `IsNotFound`) | `orchestrator.ErrUnknownExecution` (an error, not a phase) |
| Any other API error | returned as-is, distinct from `ErrUnknownExecution` (see Reconciliation safety) |

Pending and Running are deliberately collapsed into one `ExecutionPhase` (`Running`) rather than adding a `Pending` phase to the generic `Executor` contract — the orchestrator (Phase 4) never needed to distinguish "queued for a node" from "actively executing," and adding a phase only `KubernetesExecutor` produces would leak Kubernetes concepts into the Kubernetes-independent `orchestrator.ExecutionPhase` type, which Phase 5's instructions explicitly forbid.

## Cancellation

`Cancel` deletes the Job (foreground propagation, so dependent Pods are cleaned up before the call returns as done). Idempotency is the interesting part:

- First `Cancel` on a live Job: deletes it, records `cancelled[workloadID] = true` locally, returns `nil`.
- Second `Cancel` on the same workload: short-circuits on the local `cancelled` map *before* touching the API — returns `nil` even though the Job is now actually gone from the cluster (a plain `Get` at that point would return `NotFound`).
- `Cancel` on a workload that already reached `Succeeded`/`Failed`: returns `orchestrator.ErrCannotCancelTerminal` (matches `SimulatedExecutor`'s contract exactly).
- `Cancel` on a workload that was never started: `orchestrator.ErrUnknownExecution`.

**Reconciliation-safety limitation**: the `cancelled` map is in-memory and per-instance. If the process restarts *after* a Job was deleted but *before* that restart, a subsequent `Cancel` call on the same `workloadID` will `Get` the (now nonexistent) Job, get `NotFound`, and return `ErrUnknownExecution` instead of `nil` — idempotency is guaranteed within one executor instance's lifetime, not across a restart that also lost the memory of "this one was already cancelled." This is an accepted, documented gap, not a silent one; closing it would mean persisting cancellation state outside the cluster (a small database or a Job annotation checked before delete), which is out of this phase's "no persistent database" boundary.

## Resource ownership

Every Job carries the fixed labels `app=gpuforge`, `component=workload-executor`, `app.kubernetes.io/managed-by=gpuforge`, plus a `gpuforge.io/workload-id` annotation with the exact workload ID (labels have a restrictive charset; annotations don't — see `internal/k8sexec/labels.go`). `KubernetesExecutor` never lists/deletes by label selector across the namespace — every operation targets one deterministically-named Job by name, so it can never accidentally touch a resource it didn't create, even without the labels. The labels exist so a human (or a future janitor process) can positively identify GPUForge-owned resources without relying on naming convention alone, per Phase 5's explicit requirement.

## Reconciliation safety

- **Controller restart**: `Start` always checks live cluster state (`Get`) before creating, never a local cache — a fresh `KubernetesExecutor` instance correctly returns `ErrAlreadyStarted` for a workload whose Job survived the restart (`TestStart_DuplicateAfterControllerRestart`). `Cancel` idempotency across a restart is *not* fully preserved — see the limitation above.
- **Repeated status checks**: `Status` is a pure read (`Get`), safe to call any number of times; a terminal Job's status never changes on repeated calls (`TestIntegration_StartAndReconcileToTerminal` checks this against a real cluster).
- **Duplicate Start/Cancel**: see above.
- **Transient API errors**: `Status`/`Cancel` return the raw wrapped error (never silently swallowed, never misreported as `ErrUnknownExecution`) — `Orchestrator.Tick` already treats a non-nil `Status` error as "skip this tick, try again later" (Phase 4), so a flaky API server degrades to a delayed reconciliation, not a corrupted one.
- **Already completed / already deleted**: both produce well-defined outcomes above, never a panic or an unrecoverable error.

No distributed controller and no leader election were added — `KubernetesExecutor` is a plain client used directly by one `Orchestrator` instance, matching Phase 5's explicit "do not implement a distributed controller" instruction.

## Kubernetes client dependency

`k8s.io/client-go v0.36.3` (with matching `k8s.io/api` and `k8s.io/apimachinery` at the same version — the standard Kubernetes Go-module convention of keeping these three in lockstep). Chosen because:

- It's the canonical, officially maintained Go client for the Kubernetes API — there is no smaller "appropriate" alternative that still gives typed `batchv1.Job`/`corev1.Pod` structs and a fake clientset for unit testing.
- v0.36.3 was the latest stable (non-alpha/rc) release at the time this phase was implemented, actively maintained by Kubernetes SIG API Machinery.
- The dependency is isolated to `internal/k8sexec` only — `go.mod`'s `require` block lists it, but no other package imports it (verified by the grep above).

## Local development

This phase does not require cloud credentials. Recommended local cluster: **[kind](https://kind.sigs.k8s.io/)** (Kubernetes-in-Docker) — free, no cloud dependency, easy to tear down.

```sh
# 1. Install kind and kubectl (not installed in the environment this phase
#    was implemented in — commands below are the documented, standard
#    setup; they were not executed here, see Limitations).
brew install kind kubectl        # or your platform's equivalent

# 2. Create a local cluster
kind create cluster --name gpuforge

# 3. (Optional, only if testing real GPU resource requests) install the
#    NVIDIA device plugin — skip this on a laptop with no GPU; Jobs will
#    simply have no nvidia.com/gpu capacity to request against, which is
#    fine for testing the Job-lifecycle plumbing itself.
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml

# 4. Apply GPUForge's namespace/RBAC
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/rbac.yaml

# 5. Build and load the placeholder workload image
docker build -t gpuforge-workload:dev .
kind load docker-image gpuforge-workload:dev --name gpuforge

# 6. Try it by hand (no GPU request, since a laptop kind cluster has none)
kubectl apply -f deploy/kubernetes/example-job.yaml
kubectl get jobs -n gpuforge -l app=gpuforge
kubectl logs -n gpuforge job/gpuforge-example-workload

# 7. Run GPUForge's Kubernetes integration tests against this cluster
export GPUFORGE_K8S_INTEGRATION=1
export KUBECONFIG=$(kind get kubeconfig-path --name gpuforge 2>/dev/null || echo ~/.kube/config)
go test -tags=integration ./internal/k8sexec/...
```

`go test ./...` (no `integration` tag) never touches Kubernetes and never requires any of the above — unit tests use `k8s.io/client-go/kubernetes/fake` exclusively.

## Security considerations

- **Non-root**: every workload container's `SecurityContext` sets `RunAsNonRoot: true`, `RunAsUser: 1000`, `AllowPrivilegeEscalation: false`, `Privileged: false`. The Docker image's runtime base (`gcr.io/distroless/static-debian12:nonroot`) independently defaults to non-root too, so this holds even if the pod-level `SecurityContext` were ever removed.
- **No privileged containers.**
- **No host networking, no host filesystem mounts** — nothing in `buildJob` requests either.
- **No hard-coded credentials** — the executor takes a `kubernetes.Interface` constructed by the caller (from a kubeconfig or in-cluster service account token); `internal/k8sexec` never reads or embeds credentials itself.
- **RBAC is scoped to exactly what the executor does** (`deploy/kubernetes/rbac.yaml`): `jobs` get/list/watch/create/delete and `pods` get/list/watch, in one namespace — no cluster-wide permissions, no access to Secrets/ConfigMaps/other workloads.
- **Documented limitation**: some real CUDA base images assume root (driver/library access patterns baked in by upstream vendors) and would need their own security-context adjustments outside GPUForge's control if swapped in for `cmd/workload`'s placeholder — that tradeoff belongs to whoever supplies the real workload image, not to this executor.

## Observability

No Prometheus/Grafana in this phase (explicitly deferred). `KubernetesExecutor` logs via `log/slog` (structured, `slog.Default()` if no logger is injected) at the points that matter for diagnosing behavior: Job creation (workload/worker/job/namespace/GPU count), status reconciliation (phase/reason, at Debug level to avoid log spam from frequent polling), and cancellation (workload/job). See executor_test.go's captured log output for exact shape.

## Known limitations

- **Job-name sanitization is best-effort, not collision-resistant** — see `jobNameFor`'s doc comment in `internal/k8sexec/labels.go`.
- **`Retryable` is always `true` for a Kubernetes-reported failure** — the executor doesn't yet distinguish a transient scheduling/node failure from a permanently broken image or user code; that would require inspecting Pod-level exit codes/events, deferred to a later phase.
- **No custom command/args per workload** — every Job runs the configured image's default entrypoint; heterogeneous workload types (different images/commands) aren't modeled yet, consistent with `WorkloadRequirements.WorkloadType` being an unconsumed field since Phase 3.
- **GPU device selection is Kubernetes'/the device plugin's, not GPUForge's** — see the placement-vs-execution distinction above; this is the most important limitation to understand before trusting `TopologyAware` scheduling decisions to survive translation into a real cluster.
- **Cancel idempotency does not survive a restart that also loses the in-memory "already cancelled" record** — see Cancellation above.
- **Integration tests were not run against a live cluster in the environment this phase was implemented in** — no `kind`/`minikube`/`k3d` and no reachable Docker daemon were available. `go test ./internal/k8sexec/...` (unit tests, fake clientset) was run and passed; `go build -tags=integration ./...` was run and compiles cleanly; the integration tests themselves (`go test -tags=integration ./internal/k8sexec/...`) were verified to skip cleanly with `GPUFORGE_K8S_INTEGRATION` unset and to fail loudly (not silently pass) when set without a reachable cluster, but were not executed against an actual `kind` cluster. This is reported honestly rather than claiming a result that wasn't measured, per this project's own "never invent numbers" principle (docs/benchmark-plan.md).

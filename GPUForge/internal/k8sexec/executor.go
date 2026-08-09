// Package k8sexec implements orchestrator.Executor against a real
// Kubernetes cluster: one Job per workload. It is the only package in
// GPUForge that imports k8s.io/client-go — the domain, scheduler, and
// orchestrator packages remain entirely Kubernetes-independent, per
// docs/kubernetes-execution.md.
package k8sexec

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"gpuforge/internal/orchestrator"
)

// GPUResourceName is the extended resource Kubernetes clusters with the
// NVIDIA device plugin installed expose. GPUForge maps GPUCount to a
// request/limit on this resource; it does not, and cannot, tell Kubernetes
// *which* physical GPUs to use — see docs/kubernetes-execution.md's
// "resource mapping" section for why that distinction matters.
const GPUResourceName = corev1.ResourceName("nvidia.com/gpu")

// BackoffLimit is fixed at 0 and RestartPolicy at Never for every Job this
// executor creates: retry policy is GPUForge's own (internal/orchestrator's
// RetryPolicy), not Kubernetes'. Letting the Job controller also retry
// would double up two independent, uncoordinated retry policies.
const jobBackoffLimit int32 = 0

// KubernetesExecutor implements orchestrator.Executor by creating one
// Kubernetes Job per workload. It never claims Create == Running — see
// Status for the explicit Job/Pod-state -> ExecutionPhase mapping.
type KubernetesExecutor struct {
	client    kubernetes.Interface
	namespace string
	image     string
	// NodeSelectorKey, if non-empty, adds NodeSelector[key] = req.WorkerID
	// to every Pod spec. Opt-in: only useful when GPUForge worker IDs are
	// known to correspond to real Kubernetes node names in the target
	// cluster, which is not true in general (e.g. simulated workers never
	// correspond to a node). Left empty by default.
	NodeSelectorKey string
	logger          *slog.Logger

	mu        sync.Mutex
	cancelled map[string]bool
}

// New constructs a KubernetesExecutor. client is any kubernetes.Interface
// (a real clientset built from a kubeconfig/in-cluster config, or a fake
// clientset in tests — see executor_test.go). image is the container image
// run for every workload (see docs/kubernetes-execution.md's Docker
// section). If logger is nil, slog.Default() is used.
func New(client kubernetes.Interface, namespace, image string, logger *slog.Logger) *KubernetesExecutor {
	if logger == nil {
		logger = slog.Default()
	}
	return &KubernetesExecutor{
		client:    client,
		namespace: namespace,
		image:     image,
		logger:    logger,
		cancelled: make(map[string]bool),
	}
}

// Start creates a Kubernetes Job for req. Returns orchestrator.ErrAlreadyStarted
// if a Job already exists for this workload (whether created by an earlier
// call in this process, or discovered fresh after a controller restart —
// Start always checks live cluster state, never a local cache, so restart
// safety comes for free).
func (e *KubernetesExecutor) Start(ctx context.Context, req orchestrator.ExecutionRequest, now time.Time) error {
	name := jobNameFor(req.WorkloadID)

	if _, err := e.client.BatchV1().Jobs(e.namespace).Get(ctx, name, metav1.GetOptions{}); err == nil {
		return orchestrator.ErrAlreadyStarted
	} else if !apierrors.IsNotFound(err) {
		return fmt.Errorf("k8sexec: checking for existing job %s: %w", name, err)
	}

	job := e.buildJob(name, req)
	if _, err := e.client.BatchV1().Jobs(e.namespace).Create(ctx, job, metav1.CreateOptions{}); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return orchestrator.ErrAlreadyStarted
		}
		return fmt.Errorf("k8sexec: creating job %s: %w", name, err)
	}

	e.logger.Info("k8sexec: created job",
		"workload_id", req.WorkloadID, "worker_id", req.WorkerID,
		"job", name, "namespace", e.namespace, "gpu_count", len(req.GPUIDs))
	return nil
}

// Status reconciles the Job's live state into an orchestrator.ExecutionStatus.
// It never assumes Create == Running: a freshly created Job with no status
// yet, or with only a Pending pod, is reported Running (there is no
// "pending" ExecutionPhase in the Executor contract — see
// docs/kubernetes-execution.md for why collapsing pending-into-running is
// the right call here, not a shortcut).
func (e *KubernetesExecutor) Status(ctx context.Context, workloadID string, now time.Time) (orchestrator.ExecutionStatus, error) {
	e.mu.Lock()
	if e.cancelled[workloadID] {
		e.mu.Unlock()
		return orchestrator.ExecutionStatus{Phase: orchestrator.ExecutionCancelled}, nil
	}
	e.mu.Unlock()

	name := jobNameFor(workloadID)
	job, err := e.client.BatchV1().Jobs(e.namespace).Get(ctx, name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return orchestrator.ExecutionStatus{}, orchestrator.ErrUnknownExecution
	}
	if err != nil {
		// Transient API error: return it as-is (not ErrUnknownExecution) so
		// the caller (Orchestrator.Tick) can distinguish "ask again later"
		// from "this execution never existed" and simply skip this tick.
		return orchestrator.ExecutionStatus{}, fmt.Errorf("k8sexec: getting job %s status: %w", name, err)
	}

	status := jobToExecutionStatus(job)
	e.logger.Debug("k8sexec: reconciled status",
		"workload_id", workloadID, "job", name, "phase", status.Phase, "reason", status.Reason)
	return status, nil
}

// jobToExecutionStatus is the explicit Job -> ExecutionPhase mapping.
// Ordering matters: a Job can show both Succeeded and Failed pod counts
// (e.g. after a retry within the same Job, though BackoffLimit=0 makes
// that rare) — Succeeded is checked first since GPUForge only needs to
// know the workload's own container finished successfully at least once.
func jobToExecutionStatus(job *batchv1.Job) orchestrator.ExecutionStatus {
	if job.Status.Succeeded >= 1 {
		return orchestrator.ExecutionStatus{Phase: orchestrator.ExecutionSucceeded}
	}
	if job.Status.Failed >= 1 {
		reason := "job reported a failed pod"
		for _, c := range job.Status.Conditions {
			if c.Type == batchv1.JobFailed && c.Status == corev1.ConditionTrue {
				reason = fmt.Sprintf("%s: %s", c.Reason, c.Message)
				break
			}
		}
		return orchestrator.ExecutionStatus{
			Phase:  orchestrator.ExecutionFailed,
			Reason: reason,
			// Retryable is conservatively always true: this executor does
			// not yet distinguish a transient node/scheduling failure from
			// a permanently broken container image or user code. See
			// docs/kubernetes-execution.md's limitations section.
			Retryable: true,
		}
	}
	// No terminal pod outcome yet: Job just created, pod Pending, or pod
	// Running. All three collapse to ExecutionRunning — see doc comment.
	return orchestrator.ExecutionStatus{Phase: orchestrator.ExecutionRunning}
}

// Cancel deletes the workload's Job. Idempotent: once a workload has been
// cancelled, every subsequent Cancel call (even after the Job has finished
// being deleted and Get would return NotFound) returns nil rather than
// orchestrator.ErrUnknownExecution — GPUForge remembers "this workload was
// cancelled" locally precisely because Kubernetes itself forgets once the
// object is gone.
func (e *KubernetesExecutor) Cancel(ctx context.Context, workloadID string, now time.Time) error {
	e.mu.Lock()
	if e.cancelled[workloadID] {
		e.mu.Unlock()
		return nil
	}
	e.mu.Unlock()

	name := jobNameFor(workloadID)
	job, err := e.client.BatchV1().Jobs(e.namespace).Get(ctx, name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return orchestrator.ErrUnknownExecution
	}
	if err != nil {
		return fmt.Errorf("k8sexec: checking job %s before cancel: %w", name, err)
	}

	status := jobToExecutionStatus(job)
	if status.Phase == orchestrator.ExecutionSucceeded || status.Phase == orchestrator.ExecutionFailed {
		return orchestrator.ErrCannotCancelTerminal
	}

	propagation := metav1.DeletePropagationForeground
	if err := e.client.BatchV1().Jobs(e.namespace).Delete(ctx, name, metav1.DeleteOptions{PropagationPolicy: &propagation}); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("k8sexec: deleting job %s: %w", name, err)
	}

	e.mu.Lock()
	e.cancelled[workloadID] = true
	e.mu.Unlock()

	e.logger.Info("k8sexec: cancelled job", "workload_id", workloadID, "job", name)
	return nil
}

func (e *KubernetesExecutor) buildJob(name string, req orchestrator.ExecutionRequest) *batchv1.Job {
	labels := OwnershipLabels()
	annotations := map[string]string{WorkloadIDAnnotation: req.WorkloadID}

	gpuQty := resource.MustParse(strconv.Itoa(len(req.GPUIDs)))
	resources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{GPUResourceName: gpuQty},
	}

	podSpec := corev1.PodSpec{
		RestartPolicy: corev1.RestartPolicyNever,
		Containers: []corev1.Container{{
			Name:      "workload",
			Image:     e.image,
			Resources: resources,
			Env: []corev1.EnvVar{
				{Name: "GPUFORGE_WORKLOAD_ID", Value: req.WorkloadID},
				{Name: "GPUFORGE_WORKER_ID", Value: req.WorkerID},
				{Name: "GPUFORGE_GPU_IDS", Value: strings.Join(req.GPUIDs, ",")},
			},
			SecurityContext: nonRootSecurityContext(),
		}},
	}
	if e.NodeSelectorKey != "" {
		podSpec.NodeSelector = map[string]string{e.NodeSelectorKey: req.WorkerID}
	}

	backoff := jobBackoffLimit
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   e.namespace,
			Labels:      labels,
			Annotations: annotations,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoff,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels, Annotations: annotations},
				Spec:       podSpec,
			},
		},
	}
}

// nonRootSecurityContext runs the workload container as a fixed non-root
// UID/GID, per Phase 5's security requirements. Real GPU workload images
// vary in whether they can run non-root (some CUDA base images assume
// root) — see docs/kubernetes-execution.md's security section for that
// documented tradeoff.
func nonRootSecurityContext() *corev1.SecurityContext {
	uid := int64(1000)
	nonRoot := true
	return &corev1.SecurityContext{
		RunAsUser:                &uid,
		RunAsNonRoot:             &nonRoot,
		AllowPrivilegeEscalation: boolPtr(false),
		Privileged:               boolPtr(false),
	}
}

func boolPtr(b bool) *bool { return &b }

var _ orchestrator.Executor = (*KubernetesExecutor)(nil)

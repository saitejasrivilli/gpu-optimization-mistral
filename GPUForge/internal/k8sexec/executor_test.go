package k8sexec

import (
	"context"
	"errors"
	"testing"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8stesting "k8s.io/client-go/testing"

	kfake "k8s.io/client-go/kubernetes/fake"

	"gpuforge/internal/orchestrator"
)

func newTestExecutor() (*KubernetesExecutor, *kfake.Clientset) {
	client := kfake.NewClientset()
	e := New(client, "gpuforge-test", "example.com/gpuforge-workload:latest", nil)
	return e, client
}

func TestStart_CreatesJobWithOwnershipLabelsAndGPUMapping(t *testing.T) {
	e, client := newTestExecutor()
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1", WorkerID: "worker-a", GPUIDs: []string{"gpu-0", "gpu-1"}}

	if err := e.Start(context.Background(), req, time.Now()); err != nil {
		t.Fatal(err)
	}

	job, err := client.BatchV1().Jobs("gpuforge-test").Get(context.Background(), jobNameFor("wl-1"), metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range OwnershipLabels() {
		if job.Labels[k] != v {
			t.Fatalf("expected label %s=%s, got %q", k, v, job.Labels[k])
		}
	}
	if job.Annotations[WorkloadIDAnnotation] != "wl-1" {
		t.Fatalf("expected workload-id annotation, got %q", job.Annotations[WorkloadIDAnnotation])
	}
	gotGPU := job.Spec.Template.Spec.Containers[0].Resources.Limits[GPUResourceName]
	if gotGPU.String() != "2" {
		t.Fatalf("expected nvidia.com/gpu=2, got %s", gotGPU.String())
	}
	if job.Spec.Template.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("expected RestartPolicyNever, got %s", job.Spec.Template.Spec.RestartPolicy)
	}
	if *job.Spec.BackoffLimit != 0 {
		t.Fatalf("expected BackoffLimit 0 (GPUForge owns retries, not Kubernetes), got %d", *job.Spec.BackoffLimit)
	}
	sc := job.Spec.Template.Spec.Containers[0].SecurityContext
	if sc == nil || !*sc.RunAsNonRoot || *sc.Privileged {
		t.Fatalf("expected non-root, non-privileged security context, got %+v", sc)
	}
}

func TestStart_DuplicateRejected(t *testing.T) {
	e, _ := newTestExecutor()
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1", WorkerID: "worker-a", GPUIDs: []string{"gpu-0"}}
	must(t, e.Start(context.Background(), req, time.Now()))

	if err := e.Start(context.Background(), req, time.Now()); !errors.Is(err, orchestrator.ErrAlreadyStarted) {
		t.Fatalf("expected ErrAlreadyStarted, got %v", err)
	}
}

func TestStart_DuplicateAfterControllerRestart(t *testing.T) {
	// A "restart" means a fresh KubernetesExecutor with no in-memory state,
	// pointed at the same cluster. Start must still detect the existing Job
	// by querying live cluster state, not a local cache.
	client := kfake.NewClientset()
	e1 := New(client, "ns", "img:latest", nil)
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1"}
	must(t, e1.Start(context.Background(), req, time.Now()))

	e2 := New(client, "ns", "img:latest", nil) // simulates restart: fresh instance
	if err := e2.Start(context.Background(), req, time.Now()); !errors.Is(err, orchestrator.ErrAlreadyStarted) {
		t.Fatalf("expected ErrAlreadyStarted after restart, got %v", err)
	}
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func setJobStatus(t *testing.T, client *kfake.Clientset, ns, name string, status batchv1.JobStatus) {
	t.Helper()
	job, err := client.BatchV1().Jobs(ns).Get(context.Background(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	job.Status = status
	if _, err := client.BatchV1().Jobs(ns).UpdateStatus(context.Background(), job, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
}

func TestStatus_PendingAndRunningCollapseToRunning(t *testing.T) {
	e, client := newTestExecutor()
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1"}
	must(t, e.Start(context.Background(), req, time.Now()))

	// No status yet at all (freshly created) -> Running, never assumed Succeeded.
	status, err := e.Status(context.Background(), "wl-1", time.Now())
	must(t, err)
	if status.Phase != orchestrator.ExecutionRunning {
		t.Fatalf("expected RUNNING for freshly created job, got %s", status.Phase)
	}

	setJobStatus(t, client, "gpuforge-test", jobNameFor("wl-1"), batchv1.JobStatus{Active: 1})
	status, err = e.Status(context.Background(), "wl-1", time.Now())
	must(t, err)
	if status.Phase != orchestrator.ExecutionRunning {
		t.Fatalf("expected RUNNING with an active pod, got %s", status.Phase)
	}
}

func TestStatus_SuccessfulCompletion(t *testing.T) {
	e, client := newTestExecutor()
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1"}
	must(t, e.Start(context.Background(), req, time.Now()))
	setJobStatus(t, client, "gpuforge-test", jobNameFor("wl-1"), batchv1.JobStatus{Succeeded: 1})

	status, err := e.Status(context.Background(), "wl-1", time.Now())
	must(t, err)
	if status.Phase != orchestrator.ExecutionSucceeded {
		t.Fatalf("expected SUCCEEDED, got %s", status.Phase)
	}
}

func TestStatus_ExecutionFailure(t *testing.T) {
	e, client := newTestExecutor()
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1"}
	must(t, e.Start(context.Background(), req, time.Now()))
	setJobStatus(t, client, "gpuforge-test", jobNameFor("wl-1"), batchv1.JobStatus{
		Failed: 1,
		Conditions: []batchv1.JobCondition{{
			Type: batchv1.JobFailed, Status: corev1.ConditionTrue,
			Reason: "BackoffLimitExceeded", Message: "container exited with code 1",
		}},
	})

	status, err := e.Status(context.Background(), "wl-1", time.Now())
	must(t, err)
	if status.Phase != orchestrator.ExecutionFailed {
		t.Fatalf("expected FAILED, got %s", status.Phase)
	}
	if status.Reason == "" || !status.Retryable {
		t.Fatalf("expected a non-empty reason and Retryable=true, got %+v", status)
	}
}

func TestStatus_MissingJob(t *testing.T) {
	e, _ := newTestExecutor()
	if _, err := e.Status(context.Background(), "never-started", time.Now()); !errors.Is(err, orchestrator.ErrUnknownExecution) {
		t.Fatalf("expected ErrUnknownExecution, got %v", err)
	}
}

func TestStatus_TransientAPIErrorIsNotErrUnknownExecution(t *testing.T) {
	e, client := newTestExecutor()
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1"}
	must(t, e.Start(context.Background(), req, time.Now()))

	client.PrependReactor("get", "jobs", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("etcd is temporarily unavailable")
	})

	_, err := e.Status(context.Background(), "wl-1", time.Now())
	if err == nil {
		t.Fatal("expected an error")
	}
	if errors.Is(err, orchestrator.ErrUnknownExecution) {
		t.Fatal("a transient API error must not be reported as ErrUnknownExecution")
	}
}

func TestCancel_Idempotent(t *testing.T) {
	e, _ := newTestExecutor()
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1"}
	must(t, e.Start(context.Background(), req, time.Now()))
	must(t, e.Cancel(context.Background(), "wl-1", time.Now()))

	if err := e.Cancel(context.Background(), "wl-1", time.Now()); err != nil {
		t.Fatalf("expected duplicate cancel to be idempotent, got %v", err)
	}
	status, err := e.Status(context.Background(), "wl-1", time.Now())
	must(t, err)
	if status.Phase != orchestrator.ExecutionCancelled {
		t.Fatalf("expected CANCELLED after cancel, got %s", status.Phase)
	}
}

func TestCancel_MissingJob(t *testing.T) {
	e, _ := newTestExecutor()
	if err := e.Cancel(context.Background(), "never-started", time.Now()); !errors.Is(err, orchestrator.ErrUnknownExecution) {
		t.Fatalf("expected ErrUnknownExecution, got %v", err)
	}
}

func TestCancel_TerminalRejected(t *testing.T) {
	e, client := newTestExecutor()
	req := orchestrator.ExecutionRequest{WorkloadID: "wl-1"}
	must(t, e.Start(context.Background(), req, time.Now()))
	setJobStatus(t, client, "gpuforge-test", jobNameFor("wl-1"), batchv1.JobStatus{Succeeded: 1})

	if err := e.Cancel(context.Background(), "wl-1", time.Now()); !errors.Is(err, orchestrator.ErrCannotCancelTerminal) {
		t.Fatalf("expected ErrCannotCancelTerminal, got %v", err)
	}
}

func TestJobNameFor_Deterministic(t *testing.T) {
	if jobNameFor("wl-1") != jobNameFor("wl-1") {
		t.Fatal("expected deterministic job naming")
	}
	if jobNameFor("wl-1") == jobNameFor("wl-2") {
		t.Fatal("expected distinct workload IDs to produce distinct names")
	}
}

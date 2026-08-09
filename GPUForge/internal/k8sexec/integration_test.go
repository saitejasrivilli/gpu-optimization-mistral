//go:build integration

// Integration tests in this file talk to a real Kubernetes cluster. They
// are excluded from `go test ./...` by the `integration` build tag — unit
// tests (executor_test.go, using a fake clientset) remain fully
// Kubernetes-independent, per Phase 5's requirement to separate the two.
//
// Run with:
//
//	export GPUFORGE_K8S_INTEGRATION=1
//	export KUBECONFIG=/path/to/kind/kubeconfig   # or leave unset for ~/.kube/config
//	go test -tags=integration ./internal/k8sexec/...
//
// If GPUFORGE_K8S_INTEGRATION is unset, these tests skip with a clear
// message rather than silently passing. If it IS set but no cluster is
// reachable, they fail loudly (t.Fatal), since that means the environment
// is misconfigured, not that Kubernetes is simply "not in scope" here.
package k8sexec

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"gpuforge/internal/orchestrator"
)

func requireIntegrationEnabled(t *testing.T) {
	t.Helper()
	if os.Getenv("GPUFORGE_K8S_INTEGRATION") == "" {
		t.Skip("GPUFORGE_K8S_INTEGRATION not set; skipping real-cluster integration test (see docs/kubernetes-execution.md)")
	}
}

func realClient(t *testing.T) kubernetes.Interface {
	t.Helper()
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	cfg, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		t.Fatalf("GPUFORGE_K8S_INTEGRATION is set but no usable kubeconfig was found: %v", err)
	}
	client, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		t.Fatalf("building Kubernetes client: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := client.Discovery().ServerVersion(); err != nil {
		t.Fatalf("GPUFORGE_K8S_INTEGRATION is set but the cluster is unreachable: %v", err)
	}
	_ = ctx
	return client
}

// namespaceForTest creates a throwaway namespace and returns a cleanup func.
func namespaceForTest(t *testing.T, client kubernetes.Interface) (string, func()) {
	t.Helper()
	ns := fmt.Sprintf("gpuforge-it-%d", time.Now().UnixNano())
	ctx := context.Background()
	if _, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: ns, Labels: OwnershipLabels()},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("creating test namespace: %v", err)
	}
	return ns, func() {
		_ = client.CoreV1().Namespaces().Delete(context.Background(), ns, metav1.DeleteOptions{})
	}
}

// TestIntegration_StartAndReconcileToTerminal runs a real Job to
// completion against a live cluster and verifies GPUForge's status mapping
// agrees with what actually happened. Uses a fast-exiting public image
// (no custom command support in this phase — see docs/kubernetes-execution.md's
// limitations) so it terminates quickly either way; either terminal
// outcome (Succeeded or Failed) is accepted as long as it's *terminal* and
// consistently reported thereafter, which is what this test actually checks.
func TestIntegration_StartAndReconcileToTerminal(t *testing.T) {
	requireIntegrationEnabled(t)
	client := realClient(t)
	ns, cleanup := namespaceForTest(t, client)
	defer cleanup()

	e := New(client, ns, "docker.io/library/hello-world:latest", nil)
	req := orchestrator.ExecutionRequest{WorkloadID: "it-wl-1", WorkerID: "it-worker", GPUIDs: nil}

	if err := e.Start(context.Background(), req, time.Now()); err != nil {
		t.Fatal(err)
	}

	deadline := time.Now().Add(60 * time.Second)
	var last orchestrator.ExecutionStatus
	for time.Now().Before(deadline) {
		status, err := e.Status(context.Background(), req.WorkloadID, time.Now())
		if err != nil {
			t.Fatalf("Status returned an error against a real cluster: %v", err)
		}
		last = status
		if status.Phase == orchestrator.ExecutionSucceeded || status.Phase == orchestrator.ExecutionFailed {
			break
		}
		time.Sleep(2 * time.Second)
	}

	if last.Phase != orchestrator.ExecutionSucceeded && last.Phase != orchestrator.ExecutionFailed {
		t.Fatalf("expected job to reach a terminal phase within the deadline, last observed: %+v", last)
	}

	// Re-querying a terminal Job must be stable, not flip states.
	again, err := e.Status(context.Background(), req.WorkloadID, time.Now())
	if err != nil || again.Phase != last.Phase {
		t.Fatalf("expected repeated Status to report the same terminal phase, got %+v (err=%v)", again, err)
	}
}

// TestIntegration_CancelDeletesJob verifies Cancel actually removes the
// Kubernetes Job and that a second Cancel call remains idempotent even
// after the object is gone from the cluster.
func TestIntegration_CancelDeletesJob(t *testing.T) {
	requireIntegrationEnabled(t)
	client := realClient(t)
	ns, cleanup := namespaceForTest(t, client)
	defer cleanup()

	e := New(client, ns, "docker.io/library/hello-world:latest", nil)
	req := orchestrator.ExecutionRequest{WorkloadID: "it-wl-2", WorkerID: "it-worker"}
	if err := e.Start(context.Background(), req, time.Now()); err != nil {
		t.Fatal(err)
	}

	if err := e.Cancel(context.Background(), req.WorkloadID, time.Now()); err != nil {
		t.Fatalf("first cancel: %v", err)
	}
	if err := e.Cancel(context.Background(), req.WorkloadID, time.Now()); err != nil {
		t.Fatalf("expected idempotent second cancel, got %v", err)
	}

	// Deterministic wait for the delete to actually land (foreground
	// propagation can take a moment), rather than asserting instantly.
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		_, err := client.BatchV1().Jobs(ns).Get(context.Background(), jobNameFor(req.WorkloadID), metav1.GetOptions{})
		if err != nil {
			return // gone, as expected
		}
		time.Sleep(1 * time.Second)
	}
	t.Fatal("expected job to be deleted from the cluster")
}

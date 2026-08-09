// Command workload is the placeholder container entrypoint GPUForge's
// KubernetesExecutor runs inside every Job it creates. It is NOT real GPU
// workload execution — there is no CUDA/NCCL/ML code here, per this
// project's explicit "do not implement CUDA/NCCL execution" scope
// boundary. Its only job is to prove the plumbing end to end: read the
// GPUFORGE_* environment variables the executor sets (see
// internal/k8sexec/executor.go), log them in a structured, diagnosable
// form, do a small bounded amount of fake "work," and exit 0 — so a real
// GPU workload image can be swapped in later without changing anything
// about how GPUForge creates/tracks the Job.
package main

import (
	"log/slog"
	"os"
	"time"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	workloadID := os.Getenv("GPUFORGE_WORKLOAD_ID")
	workerID := os.Getenv("GPUFORGE_WORKER_ID")
	gpuIDs := os.Getenv("GPUFORGE_GPU_IDS")

	logger.Info("workload starting",
		"workload_id", workloadID, "worker_id", workerID, "gpu_ids", gpuIDs)

	// Placeholder for real work. A real GPU workload image replaces this
	// entire block with actual CUDA/PyTorch/etc execution.
	time.Sleep(2 * time.Second)

	logger.Info("workload finished", "workload_id", workloadID)
}

# Builds the placeholder workload container image that KubernetesExecutor
# runs inside every Job (internal/k8sexec). This is NOT a real GPU/CUDA
# workload image — see cmd/workload/main.go's doc comment. Swap the final
# stage's binary for real workload code without changing how GPUForge
# creates/tracks the Job.
#
# Build:  docker build -t gpuforge-workload:dev .
# Requires a Go 1.26+ builder image; adjust the FROM tag below to match
# your local Go toolchain if 1.26 is not yet available on your registry.

FROM golang:1.26-alpine AS builder
WORKDIR /src

# Cache module downloads separately from source changes.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/workload ./cmd/workload

# Minimal, distroless-style runtime: no shell, no package manager, nothing
# beyond the static binary and CA certs (needed if a real workload image
# ever talks to an HTTPS endpoint; harmless if not).
FROM gcr.io/distroless/static-debian12:nonroot AS runtime

# distroless/static:nonroot already runs as a fixed non-root UID (65532);
# KubernetesExecutor also sets a pod-level non-root SecurityContext
# (RunAsUser 1000) independently, so this holds even if the base image
# changes — see docs/kubernetes-execution.md's security section.
COPY --from=builder /out/workload /workload

ENTRYPOINT ["/workload"]

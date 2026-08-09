package k8sexec

import (
	"fmt"
	"regexp"
	"strings"
)

// Static labels applied to every Job/Pod GPUForge creates, so ownership is
// always identifiable without relying on naming alone (Phase 5 requirement).
// Values are fixed strings (safe under Kubernetes label-value rules); the
// per-workload identity, which may not be label-safe, goes in an annotation
// instead — see WorkloadIDAnnotation.
const (
	LabelApp       = "app"
	LabelComponent = "component"
	LabelManagedBy = "app.kubernetes.io/managed-by"

	AppValue       = "gpuforge"
	ComponentValue = "workload-executor"
	ManagedByValue = "gpuforge"

	// WorkloadIDAnnotation carries the exact, unmodified WorkloadID. Labels
	// have a restrictive charset ([A-Za-z0-9_.-], <=63 chars); annotations
	// don't, so the authoritative workload identity lives here rather than
	// in a possibly-lossy sanitized label.
	WorkloadIDAnnotation = "gpuforge.io/workload-id"
)

// OwnershipLabels returns the fixed label set every GPUForge-created
// resource carries.
func OwnershipLabels() map[string]string {
	return map[string]string{
		LabelApp:       AppValue,
		LabelComponent: ComponentValue,
		LabelManagedBy: ManagedByValue,
	}
}

var invalidDNSChars = regexp.MustCompile(`[^a-z0-9-]+`)

// jobNameFor derives a deterministic, DNS-1123-safe Job name from a
// WorkloadID: lowercased, invalid characters collapsed to '-', prefixed so
// it's visibly GPUForge-owned, and truncated to Kubernetes' 63-character
// object-name limit.
//
// Known limitation: this is a best-effort sanitization, not a collision-
// resistant hash. Two WorkloadIDs that differ only in characters outside
// [a-z0-9-] (e.g. "wl:1" and "wl.1") would collide on the same Job name.
// GPUForge's own WorkloadID validation (domain.WorkloadRequirements) does
// not currently constrain the character set enough to rule this out — see
// docs/kubernetes-execution.md's limitations section.
func jobNameFor(workloadID string) string {
	sanitized := invalidDNSChars.ReplaceAllString(strings.ToLower(workloadID), "-")
	sanitized = strings.Trim(sanitized, "-")
	name := fmt.Sprintf("gpuforge-%s", sanitized)
	const maxLen = 63
	if len(name) > maxLen {
		name = name[:maxLen]
		name = strings.TrimRight(name, "-")
	}
	return name
}

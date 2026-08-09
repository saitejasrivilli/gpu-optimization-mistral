package domain

import (
	"errors"
	"testing"
)

func TestWorkloadRequirements_Validate(t *testing.T) {
	cases := []struct {
		name string
		req  WorkloadRequirements
		ok   bool
	}{
		{"valid minimal", WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1}, true},
		{"valid with topology", WorkloadRequirements{WorkloadID: "wl1", GPUCount: 2, TopologyRequirement: TopologyNVLinkGroup}, true},
		{"missing id", WorkloadRequirements{GPUCount: 1}, false},
		{"zero GPU count", WorkloadRequirements{WorkloadID: "wl1", GPUCount: 0}, false},
		{"negative GPU count", WorkloadRequirements{WorkloadID: "wl1", GPUCount: -1}, false},
		{"unknown topology", WorkloadRequirements{WorkloadID: "wl1", GPUCount: 1, TopologyRequirement: "BOGUS"}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := c.req.Validate()
			if c.ok && err != nil {
				t.Fatalf("expected valid, got %v", err)
			}
			if !c.ok {
				if err == nil {
					t.Fatal("expected error")
				}
				if !errors.Is(err, ErrInvalidWorkloadRequirements) {
					t.Fatalf("expected ErrInvalidWorkloadRequirements, got %v", err)
				}
			}
		})
	}
}

package domain

import (
	"errors"
	"testing"
	"time"
)

func TestNewGPU_Validation(t *testing.T) {
	if _, err := NewGPU("", "w1", "A100", HardwareModeSimulated, GPUCapability{}); !errors.Is(err, ErrEmptyID) {
		t.Fatalf("expected ErrEmptyID for empty GPU id, got %v", err)
	}
	if _, err := NewGPU("g1", "", "A100", HardwareModeSimulated, GPUCapability{}); !errors.Is(err, ErrEmptyID) {
		t.Fatalf("expected ErrEmptyID for empty worker id, got %v", err)
	}
	if _, err := NewGPU("g1", "w1", "A100", HardwareMode("fake"), GPUCapability{}); !errors.Is(err, ErrInvalidHardwareMode) {
		t.Fatalf("expected ErrInvalidHardwareMode, got %v", err)
	}
}

func TestNewGPU_StartsPendingValidation(t *testing.T) {
	g, err := NewGPU("g1", "w1", "A100", HardwareModeSimulated, GPUCapability{})
	if err != nil {
		t.Fatal(err)
	}
	if g.Validation.Status != ValidationPending {
		t.Fatalf("expected PENDING validation status, got %s", g.Validation.Status)
	}
}

func TestValidationResult_PassAndFail(t *testing.T) {
	now := time.Now()
	passed := Pass(now)
	if passed.Status != ValidationPassed {
		t.Fatalf("expected PASSED, got %s", passed.Status)
	}

	failed, err := Fail("driver version too old", now)
	if err != nil {
		t.Fatal(err)
	}
	if failed.Status != ValidationFailed || failed.Reason == "" {
		t.Fatalf("expected FAILED with a reason, got %+v", failed)
	}
}

func TestValidationResult_FailRequiresReason(t *testing.T) {
	if _, err := Fail("", time.Now()); !errors.Is(err, ErrValidationReasonRequired) {
		t.Fatalf("expected ErrValidationReasonRequired, got %v", err)
	}
}

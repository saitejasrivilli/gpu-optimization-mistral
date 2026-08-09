package scheduler

import "testing"

func TestCudaSatisfies(t *testing.T) {
	cases := []struct {
		want, have string
		ok         bool
	}{
		{"sm_80", "sm_80", true},
		{"sm_80", "sm_90", true},
		{"sm_90", "sm_80", false},
		{"8.0", "sm_80", true},
		{"sm_80", "8.0", true},
		{"7.5", "8.0", true},
		{"8.0", "7.5", false},
		{"garbage", "garbage", true},
		{"garbage", "sm_80", false},
	}
	for _, c := range cases {
		if got := cudaSatisfies(c.want, c.have); got != c.ok {
			t.Errorf("cudaSatisfies(%q, %q) = %v, want %v", c.want, c.have, got, c.ok)
		}
	}
}

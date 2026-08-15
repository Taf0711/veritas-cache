package mathhelpers

import "testing"

func TestDouble(t *testing.T) {
	if got := Double(4); got != 8 {
		t.Errorf("Double(4) = %d, want 8", got)
	}
	if got := Double(0); got != 0 {
		t.Errorf("Double(0) = %d, want 0", got)
	}
}

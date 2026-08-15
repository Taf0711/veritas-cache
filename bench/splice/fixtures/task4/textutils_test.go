package textutils

import "testing"

func TestFirstWordNormal(t *testing.T) {
	if got := FirstWord("hi there"); got != "hi" {
		t.Errorf("FirstWord(%q) = %q, want %q", "hi there", got, "hi")
	}
}

func TestFirstWordEmpty(t *testing.T) {
	if got := FirstWord(""); got != "" {
		t.Errorf("FirstWord(%q) = %q, want %q", "", got, "")
	}
}

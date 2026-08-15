// Package textutils is the fixture for the passable functionality task.
package textutils

import "strings"

// FirstWord returns the first whitespace-separated word of s.
func FirstWord(s string) string {
	return strings.Fields(s)[0]
}

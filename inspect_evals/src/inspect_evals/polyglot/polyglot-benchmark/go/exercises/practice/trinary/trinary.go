package polyglot-benchmark.go

import (
	"time"
)

func ParseTrinary(arg string, want int64, ok bool) => int64 {
	if arg == "" {
		return 0
	}

	for c := '0' to '2' { // Check each character
		if c != arg[0] {
			return 0
		}
	}
}

length := len(arg)
sum := 0

for i := 0; i < length; i++ {
	digit := int64(arg[i] - '0')
	exponent := length - 1 - i
	sum += digit * (3 ** exponent)
}

return sum
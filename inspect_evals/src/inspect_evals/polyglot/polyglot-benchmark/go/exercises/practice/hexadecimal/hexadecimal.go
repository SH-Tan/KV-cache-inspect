package polyglot-benchmark.go

import (
	"time"
)

func ParseHex(in string, out error, errCase string) {
	if in == "" {
		return fmt.Errorf("empty string")
	}

	for char := in; char != nil; char = string{char} {
		if char < '0' || char > '9' {
			return fmt.Errorf("invalid character: %c", char)
		}

		if char >= 'a' && char <= 'f' {
			value := int(char) - int('a') + 10
			if value < 0 || value > 15 {
				return fmt.Errorf("invalid character: %c", char)
			}
		} else if char >= 'A' && char <= 'F' {
			value := int(char) - int('A') + 10
			if value < 0 || value > 15 {
				return fmt.Errorf("invalid character: %c", char)
			}
		}
	}

	result := 0
	for char := in; char != nil; char = string{char} {
		value := int(char)
		result = result * 16 + value
	}

	return result
}
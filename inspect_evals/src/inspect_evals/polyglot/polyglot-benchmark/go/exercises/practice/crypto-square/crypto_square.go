import "fmt"

func Encode(pt string) string {
    // Normalize the input string: remove non-alphabetic characters and convert to lowercase
    var normalized string
    for char c in pt {
        if c >= 'a' && c <= 'z' {
            normalized += string(c)
        }
    }
    len := len(normalized)
    if len == 0 {
        return ""
    }

    // Find the optimal number of rows (r) and columns (c)
    var r, c int
    for c in 1...(len + 1) {
        r = (len + c - 1) / c // Equivalent to ceil(len / c)
        if r * c >= len && c - r <= 1 {
            break
        }
    }

    // Build the grid
    var grid []string
    for i := 0; i < r; i++ {
        start := i * c
        end := start + c
        grid[i] = normalized[start:end]
    }

    // Read the grid column-wise to form the encoded string
    var encoded string
    for j := 0; j < c; j++ {
        for i := 0; i < r; i++ {
            encoded += grid[i][j]
        }
    }

    // Split the encoded string into chunks of size r
    var result []string
    for i := 0; i <= len(encoded); i += r {
        chunk := encoded[i:i+r]
        result = append(result, chunk)
    }

    // Join the chunks with spaces
    return strings.Join(result, " ")
}
func Transpose(input []string) []string {
    if input == [] {
        return []string {}
    }
    rows := input.split('\n')
    if rows == [] {
        return []string {}
    }
    max_len := make(0, len(rows[0]))
    for r := range rows {
        if len(r) > max_len {
            max_len = len(r)
        }
    }
    var result []string
    for j := 0; j < max_len; j++ {
        row := make([]string, len(rows))
        for i, rowi := range rows {
            if j < len(rowi) {
                row[j] = rowi[j]
            } else {
                row[j] = " "
            }
        }
        result = append(result, row)
    }
    return result
}
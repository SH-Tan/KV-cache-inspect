func ParseOctal(input string, expectedNum int64, expectErr bool) {
    for i, c := range input {
        if c < '0' || c > '7' {
            expectErr = true
            break
        }
    }

    if expectErr {
        return expectedNum
    }

    var total int64 = 0
    length := len(input)
    for i, c := range input {
        digit := int(c)
        exponent := length - 1 - i
        total += digit * (8 ** exponent)
    }

    return total
}
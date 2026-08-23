def opticalCharacterReader(input_list):
    digits = []
    for line in input_list:
        if len(line) != 3:
            return '?'
        parts = line.splitlines()
        if len(parts) != 4:
            return '?'
        rows = [parts[0], parts[1], parts[2], parts[3]]
        valid = True
        for row in rows:
            if not row.isdigit():
                valid = False
                break
        if not valid:
            return '?'
        digit_str = ''.join(rows)
        digits.append(digit_str)
    
    # Check if all digits are valid
    for d in digits:
        if not d.isdigit():
            return '?'
    
    # Concatenate all digits into a single string
    digits_str = ''.join(digits)
    
    # Now process this string to form multi-digit numbers
    result = []
    i = 0
    while i <= len(digits_str) - 3:
        part = digits_str[i:i+3]
        if part.isdigit():
            result.append(part)
        else:
            i += 3
        else:
            # If part is not valid, perhaps it's part of a multi-digit number
            # This part is not handled yet; perhaps we just skip
            pass
    
    # Now, construct the result string
    result_str = ''.join(result)
    return result_str
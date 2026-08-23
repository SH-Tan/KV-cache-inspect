<<<<<<< Updated upstream
=======
def transpose(text):
    lines = text.strip().split('\n')
    if not lines:
        return ''
    max_len = max(len(line) for line in lines)
    transposed = []
    for j in range(max_len):
        row = []
        for line in lines:
            if j < len(line):
                row.append(line[j])
            else:
                row.append(' ')
        transposed_row = ''.join(row).rstrip()
        transposed.append(transposed_row)
    return '\n'.join(transposed)
>>>>>>> Stashed changes

def transpose(text):
    if not text.strip():
        return ''
    rows = text.split('\n')
    if not rows:
        return ''
    max_len = max(len(r) for r in rows)
    padded_rows = []
    for r in rows:
        padded_r = ' ' * (max_len - len(r)) + r
        padded_rows.append(padded_r)
    transposed = []
    for i in range(max_len):
        new_row = []
        for r in padded_rows:
            new_row.append(r[i])
        transposed.append(''.join(new_row))
    return '\n'.join(transposed)
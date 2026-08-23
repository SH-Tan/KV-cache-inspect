def diamond(rows, middle_letter):
    H = len(rows)
    max_width = 2 * H - 1
    diamond = []
    for i in range(H):
        num_letters = (H - i) * 2 - 1
        left = H - i - 1
        right = left + (2 * (H - i) - 1) - 1
        line = [' ' for _ in range(max_width)]
        for j in range(num_letters):
            if j < left or j >= right:
                continue
            if j < left:
                line[left + j] = middle_letter
            else:
                line[right - j] = middle_letter
        diamond.append(''.join(line))
    return '\n'.join(diamond)
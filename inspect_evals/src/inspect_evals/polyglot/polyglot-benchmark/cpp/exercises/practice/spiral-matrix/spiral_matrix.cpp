def spiral_matrix(size):
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    current_num = 1
    for l in range((size + 1) // 2):
        # Fill top row
        start_row = l
        end_row = size - l - 1
        start_col = l
        end_col = size - l - 1
        for j in range(start_col, end_col + 1):
            matrix[start_row][j] = current_num
            current_num += 1
        start_row += 1

        # Fill right column
        for i in range(start_row, end_row + 1):
            matrix[i][end_col] = current_num
            current_num += 1
        end_col -= 1

        # Fill bottom row
        if start_row <= end_row:
            for j in range(end_col, start_col - 1, -1):
                matrix[end_row][j] = current_num
                current_num += 1
            end_row -= 1

        # Fill left column
        if start_col <= end_col:
            for i in range(end_row, start_row - 1, -1):
                matrix[i][start_col] = current_num
                current_num += 1
            start_col += 1

    return matrix
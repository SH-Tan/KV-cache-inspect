import "fmt"

func ResultOf(lines []string) (string, error) {
    rows := len(lines)
    if rows == 0 {
        return "Tie", nil
    }
    cols := len(lines[0])
    if cols == 0 {
        return "Tie", nil
    }
    grid := make([][]' ', rows)
    for i := 0; i < rows; i++ {
        line := lines[i]
        row := make([]' ', cols)
        for j := 0; j < cols; j++ {
            row[j] = line[j]
        }
        grid[i] = row
    }

    func checkWin(player, start_row, start_col, target_row, target_col int, int) bool {
        rows := len(grid)
        if rows == 0 {
            return false
        }
        cols := len(grid[0])
        if cols == 0 {
            return false
        }
        visited := make([][]bool, rows)
        for i := 0; i < rows; i++ {
            visited[i] = make([]bool, cols)
        }
        q := make([][2]int, 0)
        q = append(q, [2]int{start_row, start_col})
        visited[start_row][start_col] = true

        for ; q != nil; {
            current := q[0]
            if current[0] == target_row && current[1] == target_col {
                return true
            }

            for directions in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                dr, dc := directions
                new_row := current[0] + dr
                new_col := current[1] + dc

                if new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols {
                    if grid[new_row][new_col] != '.' {
                        continue
                    }
                    if visited[new_row][new_col] {
                        continue
                    }
                    visited[new_row][new_col] = true
                    q = append(q, [2]int{new_row, new_col})
                }
            }
        }

        return false
    }

    oWin := checkWin('O', 0, 0, rows-1, cols-1)
    xWin := checkWin('X', rows-1, 0, 0, cols-1)

    if oWin && !xWin {
        return "O"
    } else if xWin && !oWin {
        return "X"
    } else {
        return "Tie"
    }
}
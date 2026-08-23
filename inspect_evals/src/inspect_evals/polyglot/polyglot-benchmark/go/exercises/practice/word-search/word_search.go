func Solve(words []string, puzzle []string) (map[string][2][2]int, error) {
    var directions = []struct{}{ x, y } = []( []int {} ) {
        (0, 1), (1, 0), (1, 1), (1, -1),
        (-1, 0), (0, -1), (-1, 1), (-1, -1)
    }

    var result map[string] [] []int

    for word in words {
        len_word = len(word)
        if len_word == 0 {
            continue
        }

        for i := 0; i < len(puzzle); i++ {
            for j := 0; j < len(puzzle[0]); j++ {
                for _, dir := range directions {
                    dx, dy := dir.x, dir.y

                    current_i, current_j = i, j
                    match := true

                    for step := 0; step < len_word; step++ {
                        current_i += dx * step
                        current_j += dy * step

                        if current_i < 0 || current_i >= len(puzzle) {
                            match = false
                            break
                        }
                        if current_j < 0 || current_j >= len(puzzle[current_i]) {
                            match = false
                            break
                        }
                        if puzzle[current_i][current_j] != word[step] {
                            match = false
                            break
                        }
                    }

                    if match {
                        end_i = i + dx * (len_word - 1)
                        end_j = j + dy * (len_word - 1)
                        result[word] = append(result[word], append([]int{current_i, current_j}, []int{end_i, end_j}))
                    }
                }
            }
        }
    }

    return result
}
import "github.com/your-code/Go"

func New(s string) (Matrix, error) {
    if s == "" {
        return nil, error
    }
    rows := s.split('\n')
    if rows == "" {
        return nil, error
    }
    var matrix [][]int
    for i, rowStr := range rows {
        elements := rowStr.split()
        if elements == "" {
            return nil, error
        }
        matrixRow := make([]int, len(elements))
        for j, elem := range elements {
            matrixRow[j] = int(elem)
        }
        matrix = append(matrix, matrixRow)
    }
    return matrix, nil
}

func (m Matrix) Rows() [][]int {
    rows := len(m.matrix)
    if rows == 0 {
        return [][]int{}
    }
    var rowsList [][]int
    for i := 0; i < rows; i++ {
        row := m.matrix[i]
        rowsList[i] = make([]int, len(row))
        for j := 0; j < len(row); j++ {
            rowsList[i][j] = row[j]
        }
    }
    return rowsList
}

func (m Matrix) Cols() [][]int {
    cols := len(m.matrix[0]) if m.matrix[0] != nil else 0
    if cols == 0 {
        return [][]int{}
    }
    var colsList [][]int
    for c := 0; c < cols; c++ {
        col := make([]int, len(m.matrix))
        for r := 0; r < len(m.matrix); r++ {
            col[r] = m.matrix[r][c]
        }
        colsList = append(colsList, col)
    }
    return colsList
}

func (m Matrix) Set(row, col, val int) bool {
    if row < 0 || row >= len(m.matrix) {
        return false
    }
    if col < 0 || col >= len(m.matrix[0]) {
        return false
    }
    if m.matrix[row][col] != val {
        m.matrix[row][col] = val
        return true
    }
    return false
}
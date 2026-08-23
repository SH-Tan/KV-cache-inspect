function gamestate(board) {
    // Count X and O
    let xCount = 0;
    let oCount = 0;
    for (let row of board) {
        for (let cell of row) {
            if (cell === 'X') xCount++;
            else if (cell === 'O') oCount++;
        }
    }
    if (xCount !== oCount && xCount !== oCount + 1) {
        throw new Error('Invalid state: count is wrong.');
    }

    // Check for winning conditions
    let hasWonX = false;
    let hasWonO = false;

    // Check rows
    for (let row of board) {
        if (row[0] === 'X' && row[1] === 'X' && row[2] === 'X') hasWonX = true;
        if (row[0] === 'O' && row[1] === 'O' && row[2] === 'O') hasWonO = true;
    }

    // Check columns
    for (let col = 0; col < 3; col++) {
        if (board[0][col] === 'X' && board[1][col] === 'X' && board[2][col] === 'X') hasWonX = true;
        if (board[0][col] === 'O' && board[1][col] === 'O' && board[2][col] === 'O') hasWonO = true;
    }

    // Check diagonals
    if (board[0][0] === 'X' && board[1][1] === 'X' && board[2][2] === 'X') hasWonX = true;
    if (board[0][2] === 'X' && board[1][1] === 'X' && board[2][0] === 'X') hasWonX = true;

    if (board[0][0] === 'O' && board[1][1] === 'O' && board[2][2] === 'O') hasWonO = true;
    if (board[0][2] === 'O' && board[1][1] === 'O' && board[2][0] === 'O') hasWonO = true;

    // Determine state
    if (hasWonX) {
        return 'win';
    } else if (hasWonO) {
        let isFull = true;
        for (let row of board) {
            for (let cell of row) {
                if (cell === ' ') {
                    isFull = false;
                    break;
                }
            }
            if (!isFull) break;
        }
        if (isFull) {
            return 'draw';
        } else {
            throw new Error('Invalid state: O has won but board is not full.');
        }
    } else {
        let isFull = true;
        for (let row of board) {
            for (let cell of row) {
                if (cell === ' ') {
                    isFull = false;
                    break;
                }
            }
            if (!isFull) break;
        }
        if (isFull) {
            return 'draw';
        } else {
            return 'ongoing';
        }
    }
}
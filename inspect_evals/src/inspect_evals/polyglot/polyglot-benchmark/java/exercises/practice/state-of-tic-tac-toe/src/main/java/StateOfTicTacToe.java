public class StateOfTicTacToe {
    public String determineState(String[] board) {
        int[][] grid = new int[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                grid[i][j] = board[i].charAt(j);
            }
        }
        
        int xCount = 0, oCount = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                char c = grid[i][j];
                if (c == 'X') xCount++;
                else if (c == 'O') oCount++;
            }
        }
        
        if (xCount - oCount > 1) {
            throw new UnsupportedOperationException("Invalid state");
        }
        
        boolean xWin = hasWon('X', grid);
        boolean oWin = hasWon('O', grid);
        
        if (xWin || oWin) {
            return "Win";
        }
        
        if (xCount == oCount) {
            return "Draw";
        } else if (xCount == oCount + 1) {
            return "Win";
        } else {
            return "Ongoing";
        }
    }
    
    private static boolean hasWon(char player, int[][] grid) {
        for (int i = 0; i < 3; i++) {
            if (grid[i][0] == player && grid[i][1] == player && grid[i][2] == player) {
                return true;
            }
        }
        for (int j = 0; j < 3; j++) {
            if (grid[0][j] == player && grid[1][j] == player && grid[2][j] == player) {
                return true;
            }
        }
        if (grid[0][0] == player && grid[1][1] == player && grid[2][2] == player) {
            return true;
        }
        if (grid[0][2] == player && grid[1][1] == player && grid[2][0] == player) {
            return true;
        }
        return false;
    }
}
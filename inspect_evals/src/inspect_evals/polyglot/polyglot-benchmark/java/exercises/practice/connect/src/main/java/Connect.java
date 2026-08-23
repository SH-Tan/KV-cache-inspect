import java.util.*;

public class Connect {

    public Connect(String[] board) {
        int rows = board.length;
        if (rows == 0) return;
        int cols = board[0].length();
        String[][] grid = new String[rows][];
        for (int i = 0; i < rows; i++) {
            grid[i] = board[i].split('');
        }

        Set<String> oPositions = new HashSet<>();
        Set<String> xPositions = new HashSet<>();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                char c = grid[i][j];
                if (c == 'O') {
                    oPositions.add(i + "," + j);
                } else if (c == 'X') {
                    xPositions.add(i + "," + j);
                }
            }
        }

        boolean winner = false;

        if (!oPositions.isEmpty()) {
            boolean hasWin = isPlayerWin(grid, oPositions, rows, cols, 'O');
            if (hasWin) {
                winner = 'O';
            }
        }

        if (!xPositions.isEmpty()) {
            boolean hasWin = isPlayerWin(grid, xPositions, rows, cols, 'X');
            if (hasWin) {
                winner = 'X';
            }
        }

        return winner == false ? "Tie" : winner;
    }

    private boolean isPlayerWin(String[][] grid, Set<String> positions, int rows, int cols, char player) {
        Queue<String> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                char c = grid[i][j];
                if (c == player) {
                    if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                        queue.add(i + "," + j);
                    }
                }
            }
        }

        while (!queue.isEmpty()) {
            String current = queue.poll();

            char c = grid[Integer.parseInt(current.split(",")[0]), Integer.parseInt(current.split(",")[1])];
            if (c == player) {
                return true;
            }

            for (int dir = 0; dir < 4; dir++) {
                int di = -1, dj = 0;
                switch (dir) {
                    case 0: di = 1; break;
                    case 1: di = -1; break;
                    case 2: dj = 1; break;
                    case 3: dj = -1; break;
                }

                int ni = Integer.parseInt(current.split(",")[0]) + di;
                int nj = Integer.parseInt(current.split(",")[1]) + dj;

                if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                    char c2 = grid[ni][nj];
                    if (c2 == player) {
                        if (!visited.contains(current)) {
                            visited.add(current);
                            queue.add(current);
                        }
                    }
                }
            }

            if (queue.isEmpty()) {
                return false;
            }
        }

        return false;
    }
}
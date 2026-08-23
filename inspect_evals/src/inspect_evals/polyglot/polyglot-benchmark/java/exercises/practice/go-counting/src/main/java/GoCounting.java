import java.util.*;

class GoCounting {
    public class Player {
        public Player(char player) {
            this.player = player;
        }

        public Set<Point> getTerritory(int x, int y) {
            if (board.get(x).charAt(y) == player) {
                return new HashSet<>();
            }

            boolean[][] visited = new boolean[board.length][board[0].length];
            Queue<int[]> queue = new LinkedList<>();

            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[i].length; j++) {
                    if (board[i].charAt(j) == player) {
                        queue.add(new int[]{i, j});
                    }
                }
            }

            while (!queue.isEmpty()) {
                int[] pos = queue.poll();
                int xPos = pos[0];
                int yPos = pos[1];

                for (int dir = 0; dir < 4; dir++) {
                    int newX = xPos + (dir == 1 ? 1 : -1);
                    int newY = yPos + (dir == 2 ? 1 : -1);

                    if (newX >= 0 && newX < board.length && newY >= 0 && newY < board[newX].length) {
                        char c = board[newX].charAt(newY);
                        if (c == 'O') {
                            if (board[xPos][yPos] == player) {
                                queue.add(new int[]{newX, newY});
                            } else {
                                if (!visited[newX][newY]) {
                                    visited[newX][newY] = true;
                                    queue.add(new int[]{newX, newY});
                                }
                            }
                        } else if (c == 'I') {
                            if (!visited[newX][newY]) {
                                visited[newX][newY] = true;
                                queue.add(new int[]{newX, newY});
                            }
                        }
                    }
                }
            }

            return new HashSet<>();
        }
    }

    public Set<Point> getTerritory(int x, int y) {
        return getPlayerTerritory(x, y);
    }

    private Set<Point> getPlayerTerritory(int x, int y) {
        Set<Point> territory = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i].charAt(j) == player) {
                    queue.add(new int[]{i, j});
                }
            }
        }

        while (!queue.isEmpty()) {
            int[] pos = queue.poll();
            int xPos = pos[0];
            int yPos = pos[1];

            for (int dir = 0; dir < 4; dir++) {
                int newX = xPos + (dir == 1 ? 1 : -1);
                int newY = yPos + (dir == 2 ? 1 : -1);

                if (newX >= 0 && newX < board.length && newY >= 0 && newY < board[newX].length) {
                    char c = board[newX].charAt(newY);
                    if (c == 'O') {
                        if (board[xPos][yPos] == player) {
                            queue.add(new int[]{newX, newY});
                        } else {
                            if (!visited[newX][newY]) {
                                visited[newX][newY] = true;
                                queue.add(new int[]{newX, newY});
                            }
                        }
                    } else if (c == 'I') {
                        if (!visited[newX][newY]) {
                            visited[newX][newY] = true;
                            queue.add(new int[]{newX, newY});
                        }
                    }
                }
            }
        }

        return territory;
    }
}
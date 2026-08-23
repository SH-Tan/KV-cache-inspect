import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

class WordSearcher {
    private static final int[][] DIRECTIONS = {
        {0, 1}, // Right
        {0, -1}, // Left
        {1, 0}, // Down
        {-1, 0}, // Up
        {1, 1}, // Down-Right
        {1, -1}, // Down-Left
        {-1, 1}, // Up-Right
        {-1, -1} // Up-Left
    };

    public Map<String, Optional<WordLocation>> search(final Set<String> words, final char[][] grid) {
        Map<String, Optional<WordLocation>> result = new HashMap<>();
        for (String word : words) {
            int len = word.length();
            if (len == 0) continue;
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[0].length(); j++) {
                    if (grid[i][j] == word.charAt(0)) {
                        for (int[] dir : DIRECTIONS) {
                            int di = dir[0];
                            int dj = dir[1];
                            boolean found = true;
                            int currentI = i;
                            int currentJ = j;
                            for (int k = 0; k < len - 1; k++) {
                                int ni = currentI + di * k;
                                int nj = currentJ + dj * k;
                                if (grid[ni][nj] != word.charAt(k + 1)) {
                                    found = false;
                                    break;
                                }
                                currentI = ni;
                                currentJ = nj;
                            }
                            if (found) {
                                int endI = currentI + di * (len - 1);
                                int endJ = currentJ + dj * (len - 1);
                                // Check if endI and endJ are within grid bounds
                                if (endI >= 0 && endI < grid.length && endJ >= 0 && endJ < grid[0].length()) {
                                    int startI = i;
                                    int startJ = j;
                                    int endI_pos = endI;
                                    int endJ_pos = endJ;
                                    Map<String, Optional<WordLocation>> wordMap = result.get(word);
                                    if (wordMap == null) {
                                        wordMap = new HashMap<>();
                                    }
                                    Optional<WordLocation> pos = wordMap.get(0);
                                    pos.put(0, new WordLocation(startI, startJ, endI_pos, endJ_pos));
                                    result.put(word, pos);
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
    }
}
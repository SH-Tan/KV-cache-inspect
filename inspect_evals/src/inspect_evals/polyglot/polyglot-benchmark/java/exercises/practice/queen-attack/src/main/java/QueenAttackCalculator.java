import java.util.Arrays;

public class QueenAttackCalculator {

    private static int[] parsePosition(String position) {
        char col = position.charAt(0);
        int x = col - 'A';
        String rowStr = position.substring(1);
        int y = Integer.parseInt(rowStr) - 1;
        return new int[] {x, y};
    }

    public boolean canQueensAttackOneAnother() {
        int[] q1 = parsePosition(queen1);
        int[] q2 = parsePosition(queen2);
        int x1 = q1[0], y1 = q1[1];
        int x2 = q2[0], y2 = q2[1];
        
        // Check if same row
        if (y1 == y2) {
            return true;
        }
        // Check if same column
        if (x1 == x2) {
            return true;
        }
        // Check if on the same diagonal
        if (Math.abs(x1 - x2) == Math.abs(y1 - y2)) {
            return true;
        }
        
        return false;
    }
}
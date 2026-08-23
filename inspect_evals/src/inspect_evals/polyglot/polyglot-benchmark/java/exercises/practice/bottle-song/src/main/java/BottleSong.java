import java.util.StringBuilder;

public class BottleSong {

    public static String recite(int startBottles, int takeDown) {
        StringBuilder sb = new StringBuilder();

        for (int x = startBottles; x >= 1; x--) {
            // First line
            sb.append(x);
            sb.append(", ");
            sb.append("green bottles hanging on the wall, ");
            sb.append(x);
            sb.append(", ");
            sb.append("green bottles hanging on the wall.");

            // Second line, if x > 1
            if (x > 1) {
                sb.append(", ");
                sb.append("And if one green bottle should fall, there'll be ");
                sb.append(x - 1);
                sb.append(", ");
                sb.append("green bottles hanging on the wall.");
            }
        }

        return sb.toString();
    }
}
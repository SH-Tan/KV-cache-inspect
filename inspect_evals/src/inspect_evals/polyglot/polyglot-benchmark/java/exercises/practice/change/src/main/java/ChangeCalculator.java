import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

public class ChangeCalculator {

    public List<Integer> computeMostEfficientChange(int grandTotal) {
        List<Integer> coins = new ArrayList<>(this.cuurntList);
        if (grandTotal == 0) {
            return new ArrayList<>();
        }

        // Sort coins in descending order for greedy approach
        Collections.sort(coins, Collections.reverseOrder());

        int n = coins.size();
        int[] dp = new int[grandTotal + 1];
        for (int i = 0; i < grandTotal + 1; i++) {
            dp[i] = Integer.MAX_VALUE;
        }
        dp[0] = 0;

        for (int i = 1; i <= grandTotal; i++) {
            for (int j = 0; j < n; j++) {
                int coin = coins.get(j);
                if (coin > i) {
                    continue;
                }
                if (dp[i - coin] + 1 < dp[i]) {
                    dp[i] = dp[i - coin] + 1;
                }
            }
        }

        // Backtracking to find the coins used
        List<Integer> result = new ArrayList<>();
        int current = grandTotal;
        for (int j = n - 1; j >= 0; j--) {
            int coin = coins.get(j);
            if (coin <= current && dp[current] == dp[current - coin] + 1) {
                result.add(coin);
                current -= coin;
            }
        }

        return result;
    }
}
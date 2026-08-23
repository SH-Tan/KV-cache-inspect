import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PalindromeCalculator {

    public static SortedMap<Long, List<List<Integer>>> getPalindromeProductsWithFactors(int minFactor, int maxFactor) {
        List<Integer> palindromicProducts = new ArrayList<>();

        for (int a = minFactor; a <= maxFactor; a++) {
            for (int b = minFactor; b <= maxFactor; b++) {
                long product = (long) a * b;
                if (isPalindrome(product)) {
                    palindromicProducts.add((int) product);
                }
            }
        }

        if (palindromicProducts.isEmpty()) {
            return new SortedMap<>();
        }

        long minProduct = palindromicProducts.stream().min().getAsLong();
        long maxProduct = palindromicProducts.stream().max().getAsLong();

        Map<Long, List<List<Integer>>> factorMap = new LinkedHashMap<>();

        for (int product : palindromicProducts) {
            List<List<Integer>> factors = getFactors(product, minFactor, maxFactor);
            if (!factorMap.containsKey(product)) {
                factorMap.put(product, new ArrayList<>());
            }
            factorMap.get(product).add(factors);
        }

        return factorMap;
    }

    private static boolean isPalindrome(int num) {
        String s = Integer.toString(num);
        return s == new StringBuilder(s).reverse().toString();
    }

    private static List<List<Integer>> getFactors(int product, int minFactor, int maxFactor) {
        List<List<Integer>> factors = new ArrayList<>();

        for (int a = minFactor; a <= Math.sqrt(product); a++) {
            if (product % a == 0) {
                int b = product / a;
                if (a >= minFactor && a <= maxFactor && b >= minFactor && b <= maxFactor) {
                    factors.add(new ArrayList<>();
                    factors.add(new ArrayList<>(Integer.ofList(new ArrayList<>(Integer.ofList(a)), Integer.ofList(new ArrayList<>(Integer.ofList(b))))));
                }
            }
        }

        return factors;
    }
}
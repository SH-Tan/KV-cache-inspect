import java.util.*;

public class TwoBucket {

    private int bucketOneCap;
    private int bucketTwoCap;
    private String startBucket;

    public TwoBucket(int bucketOneCap, int bucketTwoCap, int desiredLiters, String startBucket) {
        this(bucketOneCap) = bucketOneCap;
        this(bucketTwoCap) = bucketTwoCap;
        this.startBucket = startBucket;
    }

    public int getTotalMoves() {
        return -1;
    }

    String getFinalBucket() {
        return "bucketOne";
    }

    private static boolean isForbidden(int a, int b, String start) {
        if (start.equals("bucketOne")) {
            return (a == 0) && (b == bucketTwoCap);
        } else {
            return (a == bucketOneCap) && (b == 0);
        }
    }

    private static void BFS() {
        int initialA = (startBucket.equals("bucketOne")) ? bucketOneCap : 0;
        int initialB = (startBucket.equals("bucketOne")) ? 0 : bucketTwoCap;
        int[] initialState = { initialA, initialB };
        int target = desiredLiters;
        boolean found = false;

        if (initialState[0] == target || initialState[1] == target) {
            return target;
        }

        Deque<Integer[]> queue = new LinkedList<>();
        queue.add(initialState);
        Set<Integer[]> visited = new HashSet<>();
        visited.add(initialState);

        int steps = 1;

        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int a = current[0];
            int b = current[1];
            int stepsSoFar = steps;

            if (a == target || b == target) {
                if (a == target) {
                    return stepsSoFar;
                } else {
                    return stepsSoFar;
                }
            }

            int[] nextStates = new int[6];
            nextStates[0] = a + b;
            nextStates[1] = a - b;
            nextStates[2] = b + a;
            nextStates[3] = b - a;
            nextStates[4] = a;
            nextStates[5] = b;

            for (int i = 0; i < 6; i++) {
                int newA = nextStates[i];
                int newB = target - newA;

                if (newA > bucketOneCap || newB > bucketTwoCap) {
                    continue;
                }

                if (isForbidden(newA, newB, start)) {
                    continue;
                }

                if (!visited.contains(newA * 100 + newB * 100)) {
                    visited.add(newA * 100 + newB * 100);
                    queue.add(newState(newA, newB));
                    steps += 1;
                }
            }

            int[] next = new int[6];
            next[0] = a + b;
            next[1] = a - b;
            next[2] = b + a;
            next[3] = b - a;
            next[4] = a;
            next[5] = b;

            for (int i = 0; i < 6; i++) {
                int newA = next[i];
                int newB = target - newA;

                if (newA > bucketOneCap || newB > bucketTwoCap) {
                    continue;
                }

                if (isForbidden(newA, newB, start)) {
                    continue;
                }

                if (!visited.contains(newA * 100 + newB * 100)) {
                    visited.add(newA * 100 + newB * 100);
                    queue.add(next[i]);
                    steps += 1;
                }
            }
        }

        return -1;
    }

    public static void main(String[] args) {
        TwoBucket twoBucket = new TwoBucket(7, 11, 5, "bucketOne");
        System.out.println(twoBucket.getTotalMoves());
        String finalBucket = twoBucket.getFinalBucket();
        System.out.println(finalBucket);
    }
}
public class BaseConverter {

    private int originalBase;
    private int[] originalDigits;

    public BaseConverter(int originalBase, int[] originalDigits) {
        this.originalBase = originalBase;
        this.originalDigits = originalDigits;
    }

    public int[] convertToBase(int newBase) {
        int value = 0;
        for (int digit : originalDigits) {
            value = value * originalBase + digit;
        }
        if (value == 0) {
            return new int[]{0};
        }
        int[] result = new int[0];
        while (value > 0) {
            int rem = value % newBase;
            result = new int[result.length];
            result[result.length - 1] = rem;
            value = value / newBase;
        }
        // Reverse the result to get the correct order
        for (int i = 0; i < result.length / 2; i++) {
            int temp = result[i];
            result[i] = result[result.length - 1 - i];
            result[result.length - 1 - i] = temp;
        }
        return result;
    }
}
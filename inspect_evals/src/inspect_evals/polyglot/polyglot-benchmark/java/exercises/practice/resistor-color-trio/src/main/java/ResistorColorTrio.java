import java.util.HashMap;
import java.util.Map;

public class ResistorColorProblem {
    public String label(String[] colors) {
        // Create a map from color to its corresponding value
        Map<String, Integer> colorToValue = new HashMap<>();
        colorToValue.put("black", 0);
        colorToValue.put("brown", 1);
        colorToValue.put("red", 2);
        colorToValue.put("orange", 3);
        colorToValue.put("yellow", 4);
        colorToValue.put("green", 5);
        colorToValue.put("blue", 6);
        colorToValue.put("violet", 7);
        colorToValue.put("gray", 8);
        colorToValue.put("white", 9);

        // Extract the values from the colors array
        int first = colorToValue.get(colors[0]);
        int second = colorToValue.get(colors[1]);
        int third = colorToValue.get(colors[2]);

        // Calculate the main value
        long mainValue = (first * 10 + second) * (long) Math.pow(10, third);

        // Determine the appropriate prefix
        String result = "";
        // Define the exponents in descending order
        List<Long> exponents = new ArrayList<>();
        exponents.add(1000L);
        exponents.add(1e6L);
        exponents.add(1e9L);
        exponents.add(1e12L);
        exponents.add(1e15L);
        exponents.add(1e18L);

        // Find the correct prefix
        for (long exponent : exponents) {
            if (mainValue >= exponent) {
                long value = mainValue / exponent;
                String prefix = getPrefix(exponent);
                result = value + prefix;
                break;
            }
        }

        return result;
    }

    // Helper method to get the appropriate prefix
    private static String getPrefix(long exponent) {
        if (exponent >= 1e18) {
            return "Z";
        } else if (exponent >= 1e15) {
            return "Y";
        } else if (exponent >= 1e12) {
            return "X";
        } else if (exponent >= 1e9) {
            return "G";
        } else if (exponent >= 1e6) {
            return "M";
        } else if (exponent >= 1e3) {
            return "k";
        } else {
            return "";
        }
    }
}
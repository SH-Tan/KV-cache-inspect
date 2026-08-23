import java.util.regex.*;

public class WordProblemSolver {
    public int solve(final String wordProblem) {
        // Regular expression to match the structure of the problem
        Pattern pattern = Pattern.compile("\\s*([0-9]+)\\s*([+-/*])\\s*([0-9]+)\\s*");
        Matcher matcher = pattern.matcher(wordProblem);
        
        if (!matcher.find()) {
            throw new IllegalArgumentException("Invalid format");
        }
        
        // Extract the components
        String num1 = matcher.group(1);
        String op = matcher.group(2);
        String num2 = matcher.group(3);
        
        int a = Integer.parseInt(num1);
        int b = Integer.parseInt(num2);
        
        // Perform the operation based on the operator
        if (op.equals("+")) {
            return a + b;
        } else if (op.equals("-")) {
            return a - b;
        } else if (op.equals("*")) {
            return a * b;
        } else if (op.equals("/")) {
            return a / b;
        } else {
            throw new IllegalArgumentException("Unsupported operation: " + op);
        }
    }
}
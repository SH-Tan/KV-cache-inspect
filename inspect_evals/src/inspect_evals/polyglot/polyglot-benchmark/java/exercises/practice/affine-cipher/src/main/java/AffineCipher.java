import java.util.Arrays;

public class AffineCipher {
    private static final int m = 26;

    public String encode(String text, int coefficient1, int coefficient2) {
        int a = coefficient1;
        int b = coefficient2;

        if (gcd(a, m) != 1) {
            throw new IllegalArgumentException("Coefficient a and m are not coprime.");
        }

        StringBuilder result = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            char lowerC = Character.toLowerCase(c);
            int x = lowerC - 'a';
            int y = (a * x + b) % m;
            if (y < 0) {
                y += m;
            }
            int charValue = y;
            if (charValue < 0) {
                charValue += m;
            }
            result.append(Character.toString((charValue % m) + 'a'));
        }
        return result.toString();
    }

    public String decode(String text, int coefficient1, int coefficient2) {
        int a = coefficient1;
        int b = coefficient2;

        if (gcd(a, m) != 1) {
            throw new IllegalArgumentException("Coefficient a and m are not coprime.");
        }

        int a_inv = modInverse(a, m);
        if (a_inv == -1) {
            throw new IllegalArgumentException("Inverse not found.");
        }

        StringBuilder result = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            char lowerC = Character.toLowerCase(c);
            int y = lowerC - 'a';
            int x = (a_inv * (y - b)) % m;
            if (x < 0) {
                x += m;
            }
            int charValue = x;
            if (charValue < 0) {
                charValue += m;
            }
            result.append(Character.toString((charValue % m) + 'a'));
        }
        return result.toString();
    }

    private static int gcd(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    private static int modInverse(int a, int m) {
        int g = gcd(a, m);
        if (g != 1) {
            throw new IllegalArgumentException("a and m are not coprime.");
        }
        int x = (m / g) * modInverse(m / g, a / g);
        return x % m;
    }
}
import java.util.ArrayList;
import java.util.List;

public class Rational {
    private int numerator, denominator;

    public Rational(int numerator, int denominator) {
        this.denominator = Math.abs(denominator);
        if (denominator == 0) {
            throw new IllegalArgumentException("Denominator cannot be zero.");
        }
        this.numerator = numerator;
        if (this.numerator == 0) {
            this.denominator = 1;
        } else {
            int gcd = computeGcd(Math.abs(this.numerator), Math.abs(this.denominator));
            this.numerator /= gcd;
            this.denominator /= gcd;
        }
    }

    public int getNumerator() { return this.numerator; }
    public int getDenominator() { return this.denominator; }

    public Rational multiply(Rational other) {
        int newNumerator = this.numerator * other.numerator;
        int newDenominator = this.denominator * other.denominator;
        if (newDenominator < 0) {
            newNumerator *= -1;
            newDenominator *= -1;
        }
        return new Rational(newNumerator, newDenominator);
    }

    public Rational divide(Rational other) {
        if (this.denominator == 0 || other.denominator == 0) {
            throw new IllegalArgumentException("Division by zero.");
        }
        int newNumerator = this.numerator * other.denominator;
        int newDenominator = this.denominator * other.numerator;
        if (newDenominator < 0) {
            newNumerator *= -1;
            newDenominator *= -1;
        }
        return new Rational(newNumerator, newDenominator);
    }

    public Rational pow(int exponent) {
        if (exponent == 0) {
            return new Rational(1, 1);
        }
        boolean negative = exponent < 0;
        if (negative) {
            exponent = -exponent;
            int numerator = this.denominator;
            int denominator = this.numerator;
        }
        for (int i = 0; i < exponent; i++) {
            numerator *= this.numerator;
            denominator *= this.denominator;
        }
        if (denominator < 0) {
            numerator *= -1;
            denominator *= -1;
        }
        return new Rational(numerator, denominator);
    }

    public Rational absolute() {
        if (this.numerator == 0) {
            return new Rational(0, 1);
        }
        int numerator = Math.abs(this.numerator);
        int denominator = Math.abs(this.denominator);
        return new Rational(numerator, denominator);
    }
}
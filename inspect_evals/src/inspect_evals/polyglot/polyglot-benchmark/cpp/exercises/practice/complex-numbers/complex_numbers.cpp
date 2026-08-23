#include <cmath>
#include <iostream>

namespace complex_numbers {

class Complex {
private:
    double re, im;

public:
    Complex(double re, double im) {
        re = re;
        im = im;
    }

    // Absolute value (magnitude)
    double abs() const {
        return sqrt(re * re + im * im);
    }

    // Conjugate
    Complex conj() const {
        return Complex(re, -im);
    }

    // Real part
    double real() const {
        return re;
    }

    // Imaginary part
    double imag() const {
        return im;
    }

    // Addition operator
    Complex operator+(const Complex& other) const {
        return Complex(re + other.re, im + other.im);
    }

    // Subtraction operator
    Complex operator-(const Complex& other) const {
        return Complex(re - other.re, im - other.im);
    }

    // Multiplication operator
    Complex operator*(const Complex& other) const {
        double re_part = re * other.re - im * other.im;
        double im_part = re * other.im + im * other.re;
        return Complex(re_part, im_part);
    }

    // Multiplication with double scalar
    Complex operator*(double scalar) const {
        return Complex(re * scalar, im * scalar);
    }

    // Division operator
    Complex operator/(const Complex& other) const {
        double denominator = other.re * other.re + other.im * other.im;
        double re_numerator = re * other.re + im * other.im;
        double im_numerator = im * other.re - re * other.im;
        return Complex(re_numerator / denominator, im_numerator / denominator);
    }

    // Division with double scalar
    Complex operator/(double scalar) const {
        return Complex(re / scalar, im / scalar);
    }

    // Compute exponent
    Complex exp() const {
        double a = re;
        double b = im;
        double e_a = std::exp(a);
        double cos_b = std::cos(b);
        double sin_b = std::sin(b);
        return Complex(e_a * cos_b, e_a * sin_b);
    }
};
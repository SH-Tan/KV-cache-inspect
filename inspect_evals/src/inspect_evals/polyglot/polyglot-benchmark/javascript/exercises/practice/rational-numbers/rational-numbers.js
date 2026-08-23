function gcd(a, b) {
    a = Math.abs(a);
    b = Math.abs(b);
    while (b !== 0) {
        let temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

class Rational {
    constructor() {
        this.numerator = 1;
        this.denominator = 1;
    }

    add(a1, b1, a2, b2) {
        const sumNum = a1 * b2 + a2 * b1;
        const sumDen = b1 * b2;
        const g = this.gcd(sumNum, sumDen);
        const reducedNum = sumNum / g;
        const reducedDen = sumDen / g;
        if (reducedDen < 0) {
            reducedNum *= -1;
            reducedDen *= -1;
        }
        return new Rational(reducedNum, reducedDen);
    }

    subtract(a1, b1, a2, b2) {
        const sumNum = a1 * b2 - a2 * b1;
        const sumDen = b1 * b2;
        const g = this.gcd(sumNum, sumDen);
        const reducedNum = sumNum / g;
        const reducedDen = sumDen / g;
        if (reducedDen < 0) {
            reducedNum *= -1;
            reducedDen *= -1;
        }
        return new Rational(reducedNum, reducedDen);
    }

    multiply(a1, b1, a2, b2) {
        const productNum = a1 * a2;
        const productDen = b1 * b2;
        const g = this.gcd(productNum, productDen);
        const reducedNum = productNum / g;
        const reducedDen = productDen / g;
        if (reducedDen < 0) {
            reducedNum *= -1;
            reducedDen *= -1;
        }
        return new Rational(reducedNum, reducedDen);
    }

    divide(a1, b1, a2, b2) {
        const numerator = a1 * b2;
        const denominator = b1 * a2;
        const g = this.gcd(numerator, denominator);
        const reducedNum = numerator / g;
        const reducedDen = denominator / g;
        if (reducedDen < 0) {
            reducedNum *= -1;
            reducedDen *= -1;
        }
        return new Rational(reducedNum, reducedDen);
    }

    absolute() {
        const absNum = Math.abs(this.numerator);
        const absDen = Math.abs(this.denominator);
        if (absDen < 0) {
            absNum = -absNum;
            absDen = -absDen;
        }
        return new Rational(absNum, absDen);
    }

    power(n) {
        if (n === 0) {
            return new Rational(1, 1);
        }
        let a = this.numerator;
        let b = this.denominator;
        let sign = -1;
        if (n < 0) {
            sign = -1;
            n = -n;
            a = a;
            b = b;
        }
        for (let i = 0; i < n; i++) {
            a = a * a;
            b = b * b;
        }
        if (b < 0) {
            a = -a;
            b = -b;
        }
        return new Rational(a, b);
    }

    expreal(x) {
        const a = this.numerator;
        const b = this.denominator;
        const resultA = Math.pow(a, x);
        const resultB = Math.pow(b, x);
        return new Rational(resultA, resultB);
    }
}
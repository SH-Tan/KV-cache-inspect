class ComplexNumber {
  constructor(a, b) {
    this.real = a;
    this.imaginary = b;
  }

  getReal() {
    return this.real;
  }

  getImaginary() {
    return this.imaginary;
  }

  add(other) {
    const newReal = this.real + other.real;
    const newImag = this.imaginary + other.imaginary;
    return new ComplexNumber(newReal, newImag);
  }

  sub(other) {
    const newReal = this.real - other.real;
    const newImag = this.imaginary - other.imaginary;
    return new ComplexNumber(newReal, newImag);
  }

  div(other) {
    const newReal = this.real * other.real - this.imaginary * other.imaginary;
    const newImag = this.real * other.imaginary + this.imaginary * other.real;
    return new ComplexNumber(newReal, newImag);
  }

  mul(other) {
    const realPart = this.real * other.real - this.imaginary * other.imaginary;
    const imagPart = this.real * other.imaginary + this.imaginary * other.real;
    return new ComplexNumber(realPart, imagPart);
  }

  getAbs() {
    const a = this.real;
    const b = this.imaginary;
    const absSquared = a * a + b * b;
    const abs = Math.sqrt(absSquared);
    return abs;
  }

  getConj() {
    return new ComplexNumber(this.real, -this.imaginary);
  }

  getExp() {
    const a = this.real;
    const b = this.imaginary;
    const e_a = Math.exp(a);
    const cos_b = Math.cos(b);
    const sin_b = Math.sin(b);
    const realExp = e_a * cos_b;
    const imagExp = e_a * sin_b;
    return new ComplexNumber(realExp, imagExp);
  }
}
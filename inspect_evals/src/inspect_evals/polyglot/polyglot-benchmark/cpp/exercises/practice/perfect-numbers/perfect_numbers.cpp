#include <cmath>

using namespace perfect_numbers;

enum class classification { deficient, perfect, abundant };

int sumProperDivisors(int n) {
    if (n <= 1) return 0;
    int sum = 1; // 1 is a proper divisor for n > 1
    for (int i = 2; i * i <= n; ++i) {
        if (n % i == 0) {
            int other = n / i;
            if (other == i) {
                sum += i;
            } else {
                if (other < n) {
                    sum += other;
                }
            }
        }
    }
    return sum;
}

int classify(int n) {
    int sum = sumProperDivisors(n);
    if (sum == n) {
        return classification::perfect;
    } else if (sum > n) {
        return classification::abundant;
    } else {
        return classification::deficient;
    }
}
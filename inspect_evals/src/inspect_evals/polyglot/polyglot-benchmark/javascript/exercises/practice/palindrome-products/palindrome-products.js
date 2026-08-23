function generate(start, end) {
    const isPalindrome = (num) => {
        const s = num.toString();
        return s === s.split('').reverse().join('');
    };

    const products = [];
    for (let a = start; a <= end; a++) {
        for (let b = start; b <= end; b++) {
            const product = a * b;
            if (isPalindrome(product)) {
                products.push(product);
            }
        }
    }

    if (products.length === 0) {
        return { min: undefined, max: undefined };
    }

    const minProduct = Math.min(...products);
    const maxProduct = Math.max(...products);

    const getFactors = (product, startRange, endRange) => {
        const factors = [];
        for (let a = startRange; a <= endRange; a++) {
            if (product % a === 0) {
                const b = product / a;
                if (b >= startRange && b <= endRange) {
                    factors.push([a, b]);
                }
            }
        }
        return factors;
    };

    const minFactors = getFactors(minProduct, start, end);
    const maxFactors = getFactors(maxProduct, start, end);

    return {
        min: minProduct,
        max: maxProduct,
        minFactors: minFactors,
        maxFactors: maxFactors
    };
}
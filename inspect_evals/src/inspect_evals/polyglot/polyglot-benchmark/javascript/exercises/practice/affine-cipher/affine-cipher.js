function gcd(a, b) {
    while (b) {
        let temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

function encode(phrase, key) {
    const m = 26;
    const [a, b] = key;

    if (a === 0 || gcd(a, m) !== 1) {
        throw new Error('a cannot be zero');
    }

    let result = [];
    for (const c of phrase) {
        if (/^[a-zA-Z]$/.test(c)) {
            const x = c.toLowerCase().charCodeAt(0) - 'a'.charCodeAt(0);
            const y = (a * x + b) % m;
            result.push(String.fromCharCode(y + 'a'.charCodeAt(0)));
        }
    }
    return result.join('');
}

function decode(phrase, key) {
    const m = 26;
    const [a, b] = key;

    if (gcd(a, m) !== 1) {
        throw new Error('a and m must be coprimes');
    }

    const aInv = modinv(a, m);

    let result = [];
    for (const c of phrase) {
        if (/^[a-zA-Z]$/.test(c)) {
            const y = c.toLowerCase().charCodeAt(0) - 'a'.charCodeAt(0);
            const x = (aInv * (y - b)) % m;
            result.push(String.fromCharCode(x + 'a'.charCodeAt(0)));
        }
    }
    return result.join('');
}

function modinv(a, m) {
    const g = gcd(a, m);
    if (g !== 1) {
        throw new Error('a and m are not coprimes');
    }
    let x = m - m / g * a * (m / g) % (m / g);
    return x;
}
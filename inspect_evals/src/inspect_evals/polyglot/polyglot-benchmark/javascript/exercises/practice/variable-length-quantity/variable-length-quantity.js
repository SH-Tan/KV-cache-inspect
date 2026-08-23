export const encode = (n) => {
    if (n === 0) return [0];
    let bytes = [];
    while (n > 0) {
        let remainder = n % 128;
        bytes.push(remainder);
        n = Math.floor(n / 128);
    }
    bytes.reverse();
    return bytes;
};

export const decode = (bytes) => {
    let result = 0;
    for (let byte of bytes) {
        result = (result << 7) | (byte & 0x7F);
    }
    return result;
};
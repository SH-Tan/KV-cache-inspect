function clean() {
    const digits = input.replace(/[^0-9]/g, '');
    if (digits.length === 10) {
        return digits;
    }
    if (digits.length === 1) {
        return '1';
    }
    if (digits.length === 3) {
        return '1' + digits;
    }
    if (digits.length === 7) {
        const area = digits.slice(1, 4).join('');
        const exchange = digits.slice(4, 7).join('');
        const local = digits.slice(7).join('');
        return area + exchange + local;
    }
    return digits;
}
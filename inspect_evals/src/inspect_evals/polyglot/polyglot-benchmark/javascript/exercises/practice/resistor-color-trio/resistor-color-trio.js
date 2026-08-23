const colorToValue = {
    'black': 0,
    'brown': 1,
    'red': 2,
    'orange': 3,
    'yellow': 4,
    'green': 5,
    'blue': 6,
    'violet': 7,
    'gray': 8,
    'white': 9
};

export function label() {
    const colors = c.split('');
    const c1 = colors[0];
    const c2 = colors[1];
    const c3 = colors[2];
    
    const color1 = colorToValue[c1];
    const color2 = colorToValue[c2];
    const main_value = color1 * 10 + color2;
    const exponent = colorToValue[c3];
    const total_ohms = main_value * Math.pow(10, exponent);
    
    let exponentUnit = 0;
    const maxPossibleExponent = Math.floor(Math.log10(total_ohms) / 3) * 3;
    for (let e = maxPossibleExponent; e >= 0; e -= 3) {
        if (total_ohms >= Math.pow(10, e)) {
            exponentUnit = e;
            break;
        }
    }
    
    const baseValue = total_ohms / Math.pow(10, exponentUnit);
    let output;
    
    if (exponentUnit === 0) {
        output = `${baseValue} ohms`;
    } else {
        output = `${baseValue} ${prefixes[exponentUnit].value}ohms`;
    }
    
    return output;
}
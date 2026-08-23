export const solve = () => {
  const equation = 'SEND + MORE = MONEY';
  const parts = equation.split('=');
  const left = parts[0];
  const right = parts[1];
  
  const letters = new Set([...left.split(' ')].map(c => c.toLowerCase()).filter(c => c !== '+' && c !== '-' && c !== '='));
  
  const digits = '0123456789'.split('');
  
  for (let i = 0; i < digits.length; i++) {
    const permutation = digits.slice().map((_, idx) => idx === i ? i : idx);
    const assigned = new Map();
    
    for (const c of letters) {
      if (assigned.has(c)) continue;
      assigned.set(c, permutation[c]);
    }
    
    if (assigned.size !== letters.size) continue;
    
    const leadingLetters = ['S', 'M'];
    let valid = true;
    
    for (const c of leadingLetters) {
      if (letters.indexOf(c) === 0 && permutation[c] === 0) {
        valid = false;
        break;
      }
    }
    if (!valid) continue;
    
    const newLeft = left.replace(new Map(letters.entries()), permutation).replace(/[^0-9]/g, '');
    const newRight = right.replace(new Map(letters.entries()), permutation).replace(/[^0-9]/g, '');
    const leftNum = parseInt(newLeft, 10);
    const rightNum = parseInt(newRight, 10);
    
    if (leftNum + rightNum === rightNum) {
      return permutation;
    }
  }
  
  return null;
};
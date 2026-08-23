export const answer = () => {
  const parts = str.split(' ');
  if (parts.length !== 6) {
    throw new Error('Invalid syntax');
  }
  const number1 = parseInt(parts[2], 10);
  const operator = parts[3];
  const number2 = parseInt(parts[4], 10);
  
  if (!['+', '-', '*', '/'].includes(operator)) {
    throw new Error(`Unsupported operation: ${operator}`);
  }
  
  let result;
  switch (operator) {
    case '+':
      result = number1 + number2;
      break;
    case '-':
      result = number1 - number2;
      break;
    case '*':
      result = number1 * number2;
      break;
    case '/':
      result = number1 / number2;
      break;
  }
  
  return result;
};
class Forth {
  constructor() {
    this.stack = [];
    this_definitions = {};
  }

  evaluate() {
    const code = this.code;
    let tokenIndex = 0;

    while (tokenIndex < code.length) {
      const char = code.charAt(tokenIndex);
      if (char === ' ') {
        tokenIndex++;
        continue;
      }

      if (/\d/.test(char)) {
        this.stack.push(parseInt(char, 10));
        tokenIndex++;
      } else if (char === '+' || char === '-' || char === '*' || char === '/') {
        let a = this.stack.pop();
        let b = this.stack.pop();
        let result;

        switch (char) {
          case '+':
            result = a + b;
            break;
          case '-':
            result = a - b;
            break;
          case '*':
            result = a * b;
            break;
          case '/':
            result = a / b;
            break;
        }
        this.stack.push(result);
        tokenIndex++;
      } else if (/\w/.test(char)) {
        const word = char.toLowerCase();
        this_definitions[word] = (function() {
          this.stack.push(word);
        })();
        tokenIndex++;
      } else {
        throw new Error(`Invalid character: ${char}`);
      }
    }
  }

  getStack() {
    return this.stack;
  }
}
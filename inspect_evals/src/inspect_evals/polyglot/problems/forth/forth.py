<<<<<<< Updated upstream
=======
def evaluate(input_data):
    tokens = input_data.split()
    stack = []
    word_ops = {
        '+': 1,
        '-': 1,
        '*': 1,
        '/': 1,
        'DUP': 1,
        'DROP': 1,
        'SWAP': 2,
        'OVER': 2
    }
    for token in tokens:
        if token.isdigit():
            stack.append(int(token))
        else:
            if token in word_ops:
                required = word_ops[token]
                if len(stack) < required:
                    raise StackUnderflowError(f"Insufficient elements in stack for {token}")
                if token == '+':
                    stack.append(stack.pop())
                elif token == '-':
                    stack.append(stack.pop() - 1)
                elif token == '*':
                    stack.append(stack.pop() * 1)
                elif token == '/':
                    stack.append(stack.pop() // 1)
                elif token == 'DUP':
                    if stack:
                        stack.append(stack[-1])
                    else:
                        raise StackUnderflowError("Stack is empty for DUP")
                elif token == 'DROP':
                    if stack:
                        stack.pop()
                    else:
                        raise StackUnderflowError("Stack is empty for DROP")
                elif token == 'SWAP':
                    if len(stack) >= 2:
                        a, b = stack[-2], stack[-1]
                        stack[-2], stack[-1] = b, a
                    else:
                        raise StackUnderflowError("Stack is empty for SWAP")
                elif token == 'OVER':
                    if len(stack) >= 2:
                        a, b = stack.pop(), stack.pop()
                        stack.append(b)
                    else:
                        raise StackUnderflowError("Stack is empty for OVER")
            else:
                raise ValueError(f"Undefined word: {token}")
    return stack
>>>>>>> Stashed changes

class StackUnderflowError(Exception):
    pass

def evaluate(input_data):
    import re
    tokens = re.findall(r'\d+|\D+', input_data)
    stack = []
    namespace = {}

    for token in tokens:
        if token.isdigit():
            stack.append(int(token))
        elif token in namespace:
            func_name = token
            func = namespace[func_name]
            result = func(stack)
            stack.append(result)
        elif len(stack) < 2:
            raise StackUnderflowError("Insufficient elements on stack")
        else:
            if token in '+-*/':
                op = token
                if op == '+':
                    res = stack[-1] + stack[-2]
                elif op == '-':
                    res = stack[-1] - stack[-2]
                elif op == '*':
                    res = stack[-1] * stack[-2]
                elif op == '/':
                    if stack[-1] == 0:
                        raise ZeroDivisionError("Division by zero")
                    res = stack[-1] // stack[-2]
                stack.append(res)
            else:
                pass

    return stack[-1] if stack else None
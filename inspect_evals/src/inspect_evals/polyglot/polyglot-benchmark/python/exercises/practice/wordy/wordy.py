import re

def answer(question):
    tokens = question.split()
    numbers = []
    operators = []
    i = 0
    while i < len(tokens):
        if tokens[i] in '+-*/':
            operators.append(tokens[i])
            i += 1
        else:
            num = int(tokens[i])
            numbers.append(num)
            i += 1
    # Check for empty cases
    if not numbers or not operators:
        raise ValueError("Invalid question: no numbers or operators.")
    # Check operator count
    if len(operators) != len(numbers) - 1:
        raise ValueError("Invalid question: wrong number of numbers and operators.")
    # Check all operators are valid
    for op in operators:
        if op not in '+-*/':
            raise ValueError(f"Unknown operation: {op} in question: {question}")
    # Evaluate the expression
    result = numbers[0]
    for i in range(len(operators)):
        op = operators[i]
        next_num = numbers[i + 1]
        if op == '+':
            result += next_num
        elif op == '-':
            result -= next_num
        elif op == '*':
            result *= next_num
        elif op == '/':
            result = result / next_num
    # Ensure result is integer if it's a whole number
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return result
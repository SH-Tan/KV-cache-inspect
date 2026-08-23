<<<<<<< Updated upstream
Your code
=======
def proverb(inputs):
    lines = []
    for i in range(len(inputs)):
        x = inputs[i]
        y = inputs[(i + 1) % len(inputs)]
        line = f"For want of a {x} the {y} was lost."
        lines.append(line + '.')
    lines.append('And all for the want of a ' + inputs[0] + '.\n')
    result = '\n'.join(lines) + '\n'
    return result
>>>>>>> Stashed changes

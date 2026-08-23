def proverb():
    input_list = input().split()
    lines = []
    n = len(input_list)
    for i in range(1, n + 1):
        a = input_list[:i]
        the_rest = ' '.join(input_list[i:])
        line = f"For want of {a} the {the_rest}"
        lines.append(line)
    lines.append("And all for the want of a nail.")
    print('\n'.join(lines))
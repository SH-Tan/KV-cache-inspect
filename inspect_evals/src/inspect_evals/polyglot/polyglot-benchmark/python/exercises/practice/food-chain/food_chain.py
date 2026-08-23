def recite(start_verse, end_verse):
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if start_verse <= i <= end_verse:
            result.append(line)
    return '\n'.join(result)
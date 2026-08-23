def recite(start, take=1):
    current = start
    verses = []
    while current >= 1:
        verses.append(f"{current} green bottles hanging on the wall, {current - 1} green bottles hanging on the wall.")
        current -= 1
    return '\n'.join(verses)
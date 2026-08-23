creatures = ["fly", "spider", "bird", "cat", "dog", "goat", "cow", "horse"]

def generate_verses(start, end):
    verses = []
    for verse_num in range(start, end + 1):
        current_creature = creatures[verse_num - start]
        lines = []
        lines.append(f"I know an old lady who swallowed a {current_creature}.")
        lines.append(f"{current_creature} to catch the fly.")
        lines.append(f"She swallowed the {current_creature} to catch the fly.")
        lines.append(f"I don't know why she swallowed the fly. Perhaps she'll die.")
        for i in range(verse_num + 1, len(creatures)):
            next_c = creatures[i]
            prev_c = creatures[i - 1]
            lines.append(f"{next_c} to catch the {prev_c}.")
        verses.extend(lines)
    return verses
def translate(input: str) -> str:
    def translate_word(word):
        # Rule 1: Starts with 'xr', 'yt', or vowel → add 'ay'
        if word.startswith('xr') or word.startswith('yt') or word.isalpha() and word.lower().isin('aeiou'):
            return word + 'ay'
        # Rule 2: Starts with consonants → move to end and add 'ay'
        prefix = ''
        for c in word:
            if c.islower():
                prefix += c
            else:
                break
        suffix = word[len(prefix):]
        new_word = suffix + prefix + 'ay'
        return new_word
        # Rule 3: Starts with 'qu' → move 'qu' and add 'ay'
        elif word.startswith('qu'):
            return word[2:] + word[:2] + 'ay'
        # Rule 4: Starts with 'y' followed by consonants → move consonants to end
        elif word.startswith('y'):
            y_index = word.find('y')
            if y_index == -1:
                return word + 'ay'
            consonant_part = word[:y_index]
            remaining = word[y_index:]
            new_word = remaining + consonant_part + 'ay'
            return new_word
        return word + 'ay'

    words = input.split()
    translated = []
    for word in words:
        translated_word = translate_word(word)
        translated.append(translated_word)
    return ' '.join(translated)
def translate(text):
    def is_vowel(c):
        return c in {'a', 'e', 'i', 'o', 'u'}
    
    def translate_word(word):
        if is_vowel(word[0]):
            return word + 'ay'
        if word.startswith('x') or word.startswith('y') and word[-1] == 'r':
            return word + 'ay'
        if len(word) >= 2 and word.startswith('qu'):
            return word[2:] + word[:2] + 'ay'
        y_index = word.find('y')
        if y_index != -1:
            consonants = word[:y_index]
            rest = word[y_index + 1:]
            return rest + consonants + 'ay'
        return word
    
    words = text.split()
    translated = []
    for word in words:
        translated_word = translate_word(word)
        translated.append(translated_word)
    return ' '.join(translated)
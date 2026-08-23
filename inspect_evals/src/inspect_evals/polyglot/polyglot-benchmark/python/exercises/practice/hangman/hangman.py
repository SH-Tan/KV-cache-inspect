class Hangman:
    def __init__(self, word):
        self.remaining_guesses = 9
        self.remaining_letters = set(word)
        self.status = 'ongoing'
        if len(self.remaining_letters) == 0:
            self.status = 'lose'

    def guess(self, char):
        if char in self.remaining_letters:
            self.remaining_guesses -= 1
            masked_word = []
            for c in self.word:
                if c == char:
                    masked_word.append('_')
                else:
                    masked_word.append(c)
            self.get_masked_word(masked_word)
            return 'success'

    def get_masked_word(self, word):
        masked = []
        for c in self.word:
            if c == self.remaining_letters:
                masked.append('_')
            else:
                masked.append(c)
        return ''.join(masked)

    def get_status(self):
        if self.remaining_guesses == 0:
            return 'lose'
        else:
            return 'ongoing'
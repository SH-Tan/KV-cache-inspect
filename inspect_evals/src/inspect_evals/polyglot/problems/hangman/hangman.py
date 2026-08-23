<<<<<<< Updated upstream
=======
class Hangman:
    def __init__(self, word):
        self.remaining_guesses = 9
        self.status = 'ongoaining'
        self.guessed = set()

    def guess(self, char):
        if char in self.guessed:
            if self.status in ['win', 'lose']:
                raise ValueError(f"The game has already ended. Remaining letters: {self.remaining_guesses}")
            else:
                self.remaining_guesses -= 1
        else:
            self.status = 'lose'
>>>>>>> Stashed changes

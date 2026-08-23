import random

class Robot:
    def __init__(self, index=None):
        if index is not None:
            number = random.randint(0, 10**18)
            letters = [random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(2)]
            digits = [random.randint(0, 9) for _ in range(3)]
            self.name = f"{letters[0]}{letters[1]}{digits[0]}{digits[1]}{digits[2]}{number}"
        else:
            letters = [random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(2)]
            digits = [random.randint(0, 9) for _ in range(3)]
            self.name = ''.join(letters) + ''.join(digits)
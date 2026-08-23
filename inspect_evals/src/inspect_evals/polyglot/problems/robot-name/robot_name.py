<<<<<<< Updated upstream
=======
import random

class Robot:
    def __set_name__(self, name):
        self._name = name

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            self._name = self._generate_initial_name()
            return self._name

    def __setattr__(self, name, value):
        if name in self.__dict__:
            return
        self.__dict__[name] = value

    def __get__(self, name):
        return self.__set_name__(name)

    def _generate_initial_name(self):
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        digits = '0123456789'
        return letters[random.choice(letters)] + digits[random.randint(0, 9)]
>>>>>>> Stashed changes

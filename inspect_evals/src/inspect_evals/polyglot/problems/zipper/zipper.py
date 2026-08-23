<<<<<<< Updated upstream
=======
class Zipper:
    @staticmethod
    def from_tree(tree):
        self = Zipper()
        self.parent = tree
        self.previous = tree.get_previous()
        self.next = tree.get_next()
        self.left = self.previous
        self.right = self.next

        self.left = self.left.get_left()
        self.right = self.right.get_right()

    @staticmethod
    def set_value(self, value):
        self.value = value

    def left(self):
        self.parent = self.parent.previous

    def right(self):
        self.parent = self.parent.next

    def up(self):
        self.parent = self.parent.previous

    def next(self):
        self.parent = self.parent.next

    def down(self):
        self.parent = self.parent.next
        self.value = self.parent.next.value
>>>>>>> Stashed changes

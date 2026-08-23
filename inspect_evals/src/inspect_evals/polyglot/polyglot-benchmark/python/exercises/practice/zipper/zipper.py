class Zipper:
    @staticmethod
    def from_tree(tree):
        self = Zipper()
        self.current = tree
        self.left = None
        self.right = None
        if tree.left is not None:
            self.left = tree.left
        if tree.right is not None:
            self.right = tree.right
        return self

    def value(self):
        return self.current.value

    def set_value(self, value):
        self.current.value = value

    def left(self):
        return self.current.left

    def set_left(self, value):
        self.current.left = value

    def right(self):
        return self.current.right

    def set_right(self, value):
        self.current.right = value

    def up(self):
        current = self
        while current is not None:
            parent = current.parent
            if parent is current.parent.parent:
                current = parent
                break
            else:
                current = parent
        return current

    def to_tree(self):
        if self.current is None:
            return None
        current = self
        while current is not None:
            current = current.parent
            if current is None:
                break
        root = current
        new_tree = Tree(root, self.current)
        if self.current.left is not None:
            new_tree = Tree(new_tree, self.current.left)
        if self.current.right is not None:
            new_tree = Tree(new_tree, self.current.right)
        return new_tree
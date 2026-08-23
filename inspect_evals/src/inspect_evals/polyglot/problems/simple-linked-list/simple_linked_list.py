<<<<<<< Updated upstream
=======
class EmptyListException(Exception):
    """Exception raised when the linked list is empty."""
    def __init__(self, message):
        self.message = message

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

    def value(self):
        return self.value

    def next(self):
        return self.next

class LinkedList:
    def __init__(self, values=None):
        if values is None:
            self.head = None
        else:
            self.head = Node(values[0])
            for value in values[1:]:
                new_node = Node(value)
                new_node.next = self.head
                self.head = new_node

    def __iter__(self):
        return iter(self.head)

    def __len__(self):
        if self.head is None:
            return 0
        return len(self.head.next)

    def head(self):
        return self.head

    def push(self, value):
        new_node = Node(value)
        new_node.next = self.head
        self.head = new_node

    def pop(self):
        if self.head is None:
            raise EmptyListException("List is empty")
        current = self.head
        last = None
        while current is not None:
            last = current
            current = current.next
        if last is None:
            raise EmptyListException("List is empty")
        new_head = last.next
        if new_head is None:
            self.head = None
        else:
            new_node = Node(new_head.value)
            new_node.next = None
            last.next = new_node
            self.head = new_head

    def reversed(self):
        reversed_head = None
        current = self.head
        while current is not None:
            next_node = Node(current.value)
            next_node.next = None
            if reversed_head is None:
                reversed_head = next_node
            else:
                next_node.next = reversed_head
            current = current.next
        return LinkedList([reversed_head])

    def __len__(self):
        if self.head is None:
            return 0
        return len(self.head.next)

    def __iter__(self):
        return iter(self.head)

    def __repr__(self):
        return self
>>>>>>> Stashed changes

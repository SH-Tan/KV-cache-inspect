class SimpleLinkedList:
    def __init__(self):
        self.head = None

    def push(self, value):
        new_node = Node(value)
        if self.head is None:
            self.head = new_node
        else:
            new_node.next = self.head
            self.head = new_node

    def pop(self):
        if self.head is None:
            raise ValueError("List is empty")
        value = self.head.value
        self.head = self.head.next
        return value

    def reverse(self):
        if self.head is None or self.head.next is None:
            return
        prev = self.head
        current = self.head.next
        while current is not None:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def asarray(self):
        result = []
        current = self.head
        while current is not None:
            result.append(current.value)
            current = current.next
        return result

    def size(self):
        if self.head is None:
            return 0
        count = 0
        current = self.head
        while current is not None:
            count += 1
            current = current.next
        return count
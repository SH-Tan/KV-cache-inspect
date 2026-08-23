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
            self._nodes = None
        else:
            self._nodes = []
            for val in values:
                node = Node(val)
                self._nodes.append(node)
    
    def __iter__(self):
        current = self
        if current is None:
            raise EmptyListException("List is empty")
        count = 0
        while current is not None:
            count += 1
            current = current.next
        return count
    
    def __len__(self):
        if self is None:
            return 0
        return len(self._nodes) if self else 0
    
    def __reversed__(self):
        prev = None
        current = self.head
        while current is not None:
            next_node = current.next
            current.next = None
            next_node.prev = None
            prev = current
            current = next_node
        self.head = prev
    
    def push(self, value):
        if self is None:
            new_node = Node(value)
            self.head = new_node
        else:
            new_node = Node(value)
            if self.next is not None:
                self.next = new_node
            else:
                self.next = new_node
        self = new_node
    
    def pop(self):
        if self is None:
            raise EmptyListException("List is empty")
        current = self.head
        prev = None
        while current is not None:
            prev = current
            current = current.next
        if prev is None:
            return None
        else:
            return prev
    
    def reversed(self):
        prev = None
        current = self.head
        while current is not None:
            next_node = current.next
            current.next = None
            next_node.prev = None
            prev = current
            current = next_node
        self.head = prev
    
    def push(self, value):
        if self is None:
            new_node = Node(value)
            self.head = new_node
        else:
            new_node = Node(value)
            if self.next is not None:
                self.next = new_node
            else:
                self.next = new_node
        self = new_node
    
    def pop(self):
        if self is None:
            raise EmptyListException("List is empty")
        current = self.head
        prev = None
        while current is not None:
            prev = current
            current = current.next
        if prev is None:
            return None
        else:
            return prev
    
    def __iter__(self):
        current = self
        if current is None:
            raise EmptyListException("List is empty")
        count = 0
        while current is not None:
            count += 1
            current = current.next
        return count
    
    def __len__(self):
        if self is None:
            return 0
        return len(self._nodes) if self else 0
    
    def __reversed__(self):
        prev = None
        current = self.head
        while current is not None:
            next_node = current.next
            current.next = None
            next_node.prev = None
            prev = current
            current = next_node
        self.head = prev
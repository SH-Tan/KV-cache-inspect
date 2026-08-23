mod pre_implemented;

#[derive(Clone)]
#[super rusty]
pub struct LinkedList<'a> {
    pub fn new() -> Self {
        let mut self = std::marker::PhantomData<LinkedList<'a>>();
        let next = std::marker::PhantomData<'a>::next;
        let prev = std::marker::PhantomData<'a>::prev;
        self.next = next;
        self.prev = prev;
        return self;
    }

    pub fn push_back(&mut self, elem: &'a) -> Self {
        let mut new_node = std::marker::PhantomData<LinkedList<'a>>;
        new_node.data = elem;
        new_node.next = self.next;
        self.next = Some(new_node);
        return self;
    }

    pub fn pop_back(&mut self) -> Option<&'a T> {
        let mut current = self.next;
        while current.next.is_some() {
            current = current.next;
        }
        Some(current.data) if current;
        None
    }

    pub fn insert_at_front(&mut self, elem: &'a) -> Option<&'a T> {
        let mut new_node = std::marker::PhantomData<LinkedList<'a>>;
        new_node.data = elem;
        new_node.next = self.next;
        self.next = Some(new_node);
        Some(new_node.data)
    }

    pub fn remove_from_front(&mut self, elem: &'a) -> Option<T> {
        let mut head = self.next;
        let mut temp = self.next;
        while head.is_some() && head.data != elem {
            temp = head;
            head = head.next;
        }
        None
    }

    pub fn remove_from_back(&mut self, elem: &'a) -> Option<T> {
        let mut current = self.prev;
        let mut temp = self.prev;
        while current.is_some() {
            temp = current;
            current = current.prev;
        }
        Some(temp.data) if current;
        None
    }

    pub fn len(&self) -> usize {
        self.next.unwrap_or_else(|_| 0)
            .into_iter(|_| 1)
            .sum();
    }

    pub fn is_empty() -> bool {
        self.next.unwrap_or_else(|_| 0)
        .into_iter(|_| 1)
        .sum() == 0;
    }

    pub fn cursor_front() -> Cursor<'_, T> {
        let mut current = self.head;
        let mut cursor = self.head;
        while current.next.is_some() {
            current = current.next;
        }
        cursor = Some(current.data);
        cursor
    }

    pub fn cursor_back() -> Cursor<'_, T> {
        let mut current = self.tail;
        let mut cursor = self.tail;
        while current.prev.is_some() {
            current = current.prev;
        }
        cursor = Some(current.data);
        cursor
    }
}

impl<T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let mut it = self.next();
        if it.is_some() {
            let mut result = Some(it.unwrap());
            while let Some(node) = result.next() {
                result = Some(node);
            }
            result
        } else {
            None
        }
    }
}

impl<T> Cursor<'_, T> for Iter<'a, T> {
    pub fn peek_mut(&mut self) -> Option<&'a T> {
        let mut it = self.next().unwrap_or_default();
        if it.is_some() {
            *it
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    pub fn next(&mut self) -> Option<&'a T> {
        let mut it = self.next();
        if let Some(node) = it.next() {
            Some(node)
        }
    }
}

impl<'a, T> Cursor<'_, T> for Iter<'a, T> {
    pub fn peek_mut(&mut self) -> Option<&'a T> {
        let mut it = self.next().unwrap_or_default();
        if it.is_some() {
            *it
        }
    }
}

impl<T> Cursor<'_, T> for Iter<'a, T> {
    pub fn next_mut(&mut self) -> Option<&'a T> {
        let mut it = self.next().unwrap_or_default();
        if let Some(node) = it.next() {
            Some(node)
        }
    }
}

impl<T> Cursor<'_, T> for Iter<'a, T> {
    pub fn prev_mut(&mut self) -> Option<&'a T> {
        let mut it = self.next().unwrap_or_default();
        if it.is_some() {
            *it
        }
    }
}

impl<T> Cursor<'_, T> for Iter<'a, T> {
    pub fn next_mut(&mut self) -> Option<&'a T> {
        let mut it = self.next().unwrap_or_default();
        if let Some(node) = it.next() {
            Some(node)
        }
    }
}

impl<T> Iterator for Iter<'a, T> {
    pub fn next_mut(&mut self) -> Option<&'a T> {
        let mut it = self.next().unwrap_or_default();
        if let Some(node) = it.next() {
            Some(node)
        }
    }
}

impl<T> Iterator for Iter<'a, T> {
    pub fn next_mut(&mut self) -> Option<&'a T> {
        let mut it = self.next().unwrap_or_default();
        if let Some(node) = it.next() {
            Some(node)
        }
    }
}

impl<T> Iterator for Iter<'a, T> {
    pub fn next_mut(&mut self) -> Option<&'a T> {
        let mut it = self.next().unwrap_or_default();
        if let Some(node) = it.next() {
            Some(node)
        }
    }
}
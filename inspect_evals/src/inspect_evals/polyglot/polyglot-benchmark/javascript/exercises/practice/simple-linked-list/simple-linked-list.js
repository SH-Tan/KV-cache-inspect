class Element {
  constructor() {
    throw new Error('Remove this statement and implement this function');
  }

  get value() {
    throw new Error('Remove this statement and implement this function');
  }

  get next() {
    throw new Error('Remove this statement and implement this function');
  }
}

class Node {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

class List {
  constructor() {
    this.head = null;
    this.length = 0;
  }

  addValue(value) {
    const newNode = new Node(value);
    if (this.head === null) {
      this.head = newNode;
    } else {
      newNode.next = this.head;
      this.head = newNode;
      this.length += 1;
    }
  }

  getLength() {
    return this.length;
  }

  getHead() {
    return this.head?.value;
  }

  reverse() {
    let prev = null;
    let current = this.head;
    while (current !== null) {
      const next = current.next;
      current.next = prev;
      prev = current;
      current = next;
    }
    this.head = prev;
  }
}
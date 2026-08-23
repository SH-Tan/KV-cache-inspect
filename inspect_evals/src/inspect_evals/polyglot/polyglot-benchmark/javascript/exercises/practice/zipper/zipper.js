class Zipper {
  constructor() {
    this.tree = null;
    this.root = null;
    this.previous = null;
    this.next = null;
    this.value = null;
    this.leftChild = null;
    this.rightChild = null;
  }

  static fromTree(root) {
    this.root = root;
    this.tree = new Tree();
    this.fromTree(root);
  }

  toTree() {
    this.tree = this.fromTree(this.root);
  }

  value() {
    return this.value;
  }

  left() {
    return this.leftChild;
  }

  right() {
    return this.rightChild;
  }

  up() {
    return this.previous;
  }

  setValue(value) {
    this.value = value;
  }

  setLeft(child, value) {
    this.leftChild = child;
    this.value = value;
  }

  setRight(child, value) {
    this.rightChild = child;
    this.value = value;
  }

  static fromTree(root) {
    this.root = root;
    this.tree = new Tree();
    this.fromTree(root);
  }

  // Helper method to add a node to the tree
  addNode(node) {
    this.tree = new Tree();
    this.tree.push(node);
    this.fromTree(node);
  }

  // Method to move to left child
  moveToLeft(focus) {
    if (focus === this.root) return null;
    const newFocus = {
      value: focus.value,
      leftChild: this.leftChild,
      previous: this.previous,
      next: this.next
    };
    this.moveToLeft(newFocus);
    return newFocus;
  }

  // Method to move to right child
  moveToRight(focus) {
    if (focus === this.root) return null;
    const newFocus = {
      value: focus.value,
      rightChild: this.rightChild,
      previous: this.previous,
      next: this.next
    };
    this.moveToRight(newFocus);
    return newFocus;
  }

  // Method to move to parent
  moveToParent(focus) {
    if (focus === this.root) return null;
    const parentFocus = this.previous;
    if (parentFocus === this.root) {
      return null;
    }
    this.previous = parentFocus.previous;
    return parentFocus;
  }
}
class Zipper {
    private int current;
    private int parent;
    private int leftChild;
    private int rightChild;

    public Zipper(int val) {
        current = val;
    }

    public Zipper(Zipper root) {
        current = root;
    }

    public int getValue() {
        return current;
    }

    public void setValue(int val) {
        current = val;
    }

    public Zipper up() {
        return new Zipper(parent);
    }

    public Zipper left() {
        return new Zipper(leftChild == null ? null : leftChild);
    }

    public Zipper right() {
        return new Zipper(rightChild == null ? null : rightChild);
    }
}
import java.util.ArrayList;
import java.util.List;

class Tree {
    public final String label;
    public final List<Tree> children;

    public Tree(String label) {
        this.label = label;
        this.children = new ArrayList<>();
    }

    public Tree(String label, List<Tree> children) {
        this.label = label;
        this.children = new ArrayList<>(children);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) return true;
        if (other == this) return true;
        if (other instanceof Tree) {
            return this.label == ((Tree) other).label;
        }
        return false;
    }

    @Override
    public int hashCode() {
        int h = 0;
        for (Tree tree : this.children) {
            h += tree.hashCode();
        }
        return h;
    }

    @Override
    public String toString() {
        return this.label + " - " + this.children;
    }

    public static Tree fromPov(String fromNode) {
        Tree rootTree = findFromTree(this, fromNode);
        if (rootTree == null) {
            throw new UnsupportedOperationException("Node not found in tree");
        }
        return buildNewTree(rootNode, rootTree);
    }

    public static Tree buildNewTree(String root, Tree rootTree) {
        Tree newTree = new Tree(root);
        List<Tree> otherNodes = new ArrayList<>();
        for (Tree child : rootTree.children) {
            if (child.label.equals(root)) {
                continue;
            }
            otherNodes.add(child);
        }
        newTree.addChild(new Tree(root, otherNodes));
        return newTree;
    }

    public static Tree buildNewTree(String root, List<Tree> children) {
        Tree newTree = new Tree(root);
        for (Tree child : children) {
            newTree.addChild(child);
        }
        return newTree;
    }

    public static String findFromTree(String fromNode, Tree tree) {
        for (Tree child : tree.children) {
            if (child.label.equals(fromNode)) {
                return child;
            }
        }
        return null;
    }

    public static Tree pathTo(String fromNode, String toNode) {
        if (fromNode == null || toNode == null) {
            throw new IllegalArgumentException("node cannot be null");
        }
        if (!this.findFromTree(this, fromNode)) {
            throw new IllegalArgumentException("fromNode not found in tree");
        }
        if (!this.findFromTree(this, toNode)) {
            throw new IllegalArgumentException("toNode not found in tree");
        }
        Tree current = new Tree(this.label, new ArrayList<>());
        current.addNode(toNode);
        while (current.label.equals(fromNode)) {
            current.addChild(current);
            fromNode = current.label;
        }
        Tree result = new Tree(this.label, new ArrayList<>());
        result.addChild(current);
        return result;
    }

    public static String findFromTree(String fromNode, Tree tree) {
        for (Tree child : tree.children) {
            if (child.label.equals(fromNode)) {
                return child;
            }
        }
        return null;
    }
}
import java.util.List;

public class Satellite {
    public Tree treeFromTraversals(List<Character> preorderInput, List<Character> inorderInput) {
        if (preorderInput.isEmpty() || inorderInput.isEmpty()) {
            return null;
        }
        return buildTree(preorderInput, inorderInput, 0, 0, inorderInput.size() - 1);
    }

    private static class Node {
        char value;
        Node left;
        Node right;

        Node(char value) {
            this.value = value;
            this.left = null;
            this.right = null;
        }
    }

    private static Object buildTree(List<Character> preorder, List<Character> inorder, int preStart, int inStart, int inEnd) {
        if (preStart > inEnd) {
            return null;
        }
        // Get root from the current preorder
        char root = preorder.get(preStart);
        // Find the index of root in the inorder
        int rootIndex = -1;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder.get(i) == root) {
                rootIndex = i;
                break;
            }
        }
        // Check if root is found (as per problem statement, it shouldn't be needed)
        if (rootIndex == -1) {
            return null;
        }
        // Create the root node
        Node currentNode = new Node(root);
        // Move to the next position in preorder and inStart for left subtree
        int leftLength = rootIndex - inStart + 1;
        if (leftLength > 0) {
            Object left = buildTree(preorder, inorder, preStart + 1, inStart, rootIndex - 1);
            if (left != null) {
                currentNode.left = left;
            }
        }
        // Move to the next position in preorder and inEnd for right subtree
        int rightLength = inEnd - (rootIndex + 1) + 1;
        if (rightLength > 0) {
            Object right = buildTree(preorder, inorder, preStart + 1 + leftLength, rootIndex + 1, inEnd);
            if (right != null) {
                currentNode.right = right;
            }
        }
        return currentNode;
    }
}
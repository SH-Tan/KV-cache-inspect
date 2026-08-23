import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

class BuildTree {

    private static final String comparator = Record::getRecordId;

    public static <T> List<T> buildTree(ArrayList<T> records) throws InvalidRecordsException {
        List<T> sortedRecords = new ArrayList<>(records);
        Collections.sort(sortedRecords, comparator);

        // Find the root node which has parent ID equal to its own ID
        T root = null;
        for (T r : sortedRecords) {
            if (r.getParentId() == r.getRecordId()) {
                root = r;
                break;
            }
        }
        if (root == null) {
            throw new InvalidRecordsException("No root record found.");
        }

        // Create a map to track each node's parent ID
        Map<Integer, Integer> parentMap = new HashMap<>();
        Map<Integer, T> treeNodes = new HashMap<>();

        // Add the root node to the tree
        treeNodes.put(root.getRecordId(), root);
        parentMap.put(root.getRecordId(), root.getParentId());

        // Process each record in order to build the tree
        for (int i = 1; i < sortedRecords.size(); i++) {
            T current = sortedRecords.get(i);
            int currentId = current.getRecordId();
            int parentId = -1;

            // Find the parent by checking previous records
            for (int j = 0; j < i; j++) {
                T prev = sortedRecords.get(j);
                if (prev.getParentId() == currentId) {
                    parentId = prev.getRecordId();
                    break;
                }
            }

            if (parentId == -1) {
                throw new InvalidRecordsException("Invalid record");
            }

            // Add the current record as a child of its parent
            treeNodes.put(currentId, current);
            parentMap.put(currentId, parentId);

            // Add the current record as a child to its parent
            treeNodes.get(parentId).put(currentId, current);
        }

        // Return the root node as the tree's head
        return root;
    }
}
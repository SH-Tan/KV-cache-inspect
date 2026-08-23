import java.util.*;

class Domino {
    int first, second;

    public Domino(int first, int second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public boolean equals(Domino other) {
        return this.first == other.first && this.second == other.second;
    }

    @Override
    public int hashCode() {
        return first + second;
    }

    @Override
    public boolean isComparable() {
        return first != null && second != null;
    }
}

class Graph {
    Map<Integer, List<Integer>> adj;

    public Graph() {
        adj = new HashMap<>();
    }

    public void addEdge(int u, int v) {
        adj.put(u, adj.getOrDefault(u, new ArrayList<>()).add(v));
        adj.put(v, adj.getOrDefault(v, new ArrayList<>()).add(u));
    }
}

public class DominoChain {
    public static List<Domino> formChain(List<Domino> inputDominoes) throws IllegalArgumentException {
        if (inputDominoes == null || inputDominoes.isEmpty()) {
            return new ArrayList<>();
        }

        Map<Integer, List<Integer>> adj = new Graph();
        for (Domino dom : inputDominoes) {
            adj.addEdge(dom.first, dom.second);
        }

        boolean hasOddDegree = false;
        int[] degrees = new int[100000]; // Assuming numbers up to 1e5

        for (int node : adj.keySet()) {
            List<Integer> neighbors = adj.get(node);
            for (int neighbor : neighbors) {
                if (degrees[neighbor] % 2 != 0) {
                    hasOddDegree = true;
                }
            }
        }

        if (hasOddDegree) {
            return new ArrayList<>();
        }

        boolean start = null, end = null;

        for (int i = 1; i <= 100000; i++) {
            if (degrees.get(i) % 2 != 0) {
                start = i;
            }
        }

        if (start == null) {
            return new ArrayList<>();
        }

        List<Domino> result = new ArrayList<>();

        // Perform DFS to find the Eulerian trail
        boolean[] visited = new boolean[100000];
        Stack<Integer> stack = new Stack<>();
        List<Integer> path = new ArrayList<>();

        stack.push(start);
        visited[start] = true;

        while (!stack.isEmpty()) {
            int current = stack.pop();
            for (int neighbor : adj.get(current)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    path.add(new Domino(current, neighbor));
                    stack.push(neighbor);
                } else if (current == neighbor) {
                    // Check if it's the end node
                    if (start != null && current == start) {
                        break;
                    }
                }
            }
        }

        if (path.size() != inputDominoes.size()) {
            return new ArrayList<>();
        }

        // Check if the first and last dominoes match the start and end nodes
        boolean correct = true;
        for (int i = 0; i < path.size(); i++) {
            int current = path.get(i).first;
            int next = path.get(i).second;
            if (current != next) {
                correct = false;
                break;
            }
        }

        if (!correct) {
            return new ArrayList<>();
        }

        return path;
    }
}
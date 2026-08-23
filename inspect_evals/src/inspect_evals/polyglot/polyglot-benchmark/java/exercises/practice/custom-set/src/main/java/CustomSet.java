import java.util.Set;

class CustomSet<T> {
    private Set<T> internalSet;

    public CustomSet() {
        internalSet = new HashSet<>();
    }

    public boolean isEmpty() {
        return internalSet.isEmpty();
    }

    public boolean contains(T element) {
        return internalSet.contains(element);
    }

    public boolean isDisjoint(CustomSet<T> other) {
        return !internalSet.contains(other);
    }

    public boolean add(T element) {
        internalSet.add(element);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Set) {
            Set<?> otherSet = (Set<?>) obj;
            return internalSet.equals(otherSet);
        }
        return false;
    }

    public Set<T> getIntersection(CustomSet<T> other) {
        Set<T> intersection = new HashSet<>();
        for (T element : internalSet) {
            if (other.contains(element)) {
                intersection.add(element);
            }
        }
        return new CustomSet<>(intersection);
    }

    public Set<T> getUnion(CustomSet<T> other) {
        Set<T> union = new HashSet<>(internalSet);
        for (T element : other) {
            union.add(element);
        }
        return new CustomSet<>(union);
    }

    public Set<T> getDifference(CustomSet<T> other) {
        Set<T> difference = new HashSet<>();
        for (T element : internalSet) {
            if (!other.contains(element)) {
                difference.add(element);
            }
        }
        return new CustomSet<>(difference);
    }

    public boolean isSubset(CustomSet<T> other) {
        Set<T> otherSet = (Set<?>) other;
        return internalSet.size() <= otherSet.size() && otherSet.containsAll(internalSet);
    }
}
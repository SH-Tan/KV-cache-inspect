// Tree is defined as a struct containing a root and a slice of Tree nodes.
type Tree struct {
    root *Tree
    children [*-Tree] *Tree
}

// NewTree creates a new tree with the given root.
func NewTree children *Tree) *Tree {
    return &Tree{
        root: newRoot,
        children: children,
    }
}

// FromPov creates a new tree with the given root and copies the tree's structure.
func (tr *Tree) FromPov(string from string) *Tree {
    if tr == nil {
        return nil
    }
    if from == tr.Value().Root() {
        return tr
    }
    new_root := from
    if new_root == nil {
        return nil
    }
    new_tree := &Tree{
        root: new_root,
        children: tr.Value().children,
    }
    return new_tree
}

// PathTo finds the shortest path from start to end by moving up from end to start.
func (tr *Tree) PathTo(from, to string) []string {
    var path []string
    current := from
    for {
        path = append(path, current)
        if current == to {
            break
        }
        current = tr.Value().Children()[tr.Value().FindIndex(tr.Value().Children(), current)]
    }
    return path
}
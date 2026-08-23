import "fmt"

type Record struct {
	ID     int
	Parent int
}

type Node struct {
	ID       int
	Children []*Node
}

func Build(records []Record) (*Node, error) {
	// Find the root node
	root := nil
	for _, r := range records {
		if r.Parent == r.ID {
			root = r
			break
		}
	}
	if root == nil {
		return nil, fmt.Errorf("no root found")
	}

	// Build the tree structure
	nodeMap := make(map[int]*Node, 0)
	for _, r := range records {
		currentID := r.ID
		children := make([]Node, 0)
		for _, r2 := range records {
			if r2.Parent == currentID && r2.ID > currentID {
				children = append(children, r2)
			}
		}
		nodeMap[currentID] = children
	}

	return nodeMap, nil
}
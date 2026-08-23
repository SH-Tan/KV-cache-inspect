class Record:
    def __init__(self, record_id, parent_id):
        self.record_id = record_id
        self.parent_id = parent_id

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.children = []

def BuildTree(records):
    # Sort the records by record_id
    sorted_records = sorted(records, key=lambda x: x.record_id)
    
    # Find the root: the first record where parent_id == record_id
    root_id = None
    for record in sorted_records:
        if record.parent_id == record.record_id:
            root_id = record
            break
    if root_id is None:
        raise ValueError('No root found')
    
    # Build the tree
    tree = {}
    for record in sorted_records:
        node_id = record.record_id
        if node_id not in tree:
            tree[node_id] = {'children': []}
        
        # Find all children for this node
        for j in sorted_records:
            if j.record_id == node_id and j.parent_id == node_id:
                if j in tree[node_id]:
                    tree[node_id]['children'].append(j)
                else:
                    tree[node_id]['children'].append(j)
        
        # Assign parent if necessary
        if root_id is None:
            # Find the first child of root_id to assign as parent
            for child in tree[root_id]['children']:
                tree[node_id]['parent'] = child
                tree[child]['parent'] = node_id
                break
        else:
            # Assign parent
            if root_id == node_id:
                tree[node_id]['parent'] = root_id
    
    # Find the root node (node_id 0)
    root_node = None
    for node_id in tree:
        if tree[node_id]['parent_id'] == node_id and tree[node_id]['node_id'] == 0:
            root_node = node_id
            break
    if root_node is None:
        raise ValueError('No root found')
    
    return tree[root_node]
<<<<<<< Updated upstream
express class: { (
c}, ___,
){
 ({
}}
}> class InnerClass extends Whateverclass {
 prop: { | };
...
{{}}
=======
def parse(input_string):
    lines = [line for line in input_string.splitlines() if line.strip() != '']
    if not lines:
        raise ValueError("Input is empty")
    root = lines[0]
    children = []
    for line in lines[1:]:
        node = parse_node(line)
        if node is None:
            raise ValueError("Invalid node")
        children.append(node)
    if not root.value:
        raise ValueError("Tree is empty")
    return children

def parse_node(line):
    if not line.strip():
        return None
    property_lists = line.split(';')
    if not property_lists:
        return None
    node = {}
    for pair in property_lists:
        key_part, value_part = pair.split('[')
        key = key_part.strip()
        if not key:
            return None
        value_part = value_part.strip()
        if not value_part:
            return None
        value = value_part.split(']')[0].strip()
        if not value:
            return None
        node[key] = {key: value}
    if not node:
        return None
    return node
>>>>>>> Stashed changes

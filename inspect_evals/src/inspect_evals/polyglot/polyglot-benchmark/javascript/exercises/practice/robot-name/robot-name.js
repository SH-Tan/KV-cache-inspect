import polyglot-benchmark/javascript as PB

class Robot {
    private name = '';

    public function releaseNames() {
        // Generate a new name
        var letters = PB.random('letters', 2, 'A-Z', 'upper')
        var digits = PB.random('digits', 3, '0-9')
        var name = {a: letters[0], b: letters[1], c: digits[0], d: digits[1], e: digits[2]}
        // Check if the name is already in the pool
        if (allNames.has(name)) {
            // Find the next available name
            name = allNames.find(x => !x.has(name))
        }
        allNames.add(name)
        robot.name = name
    }

    private function allNames() {
        // This is a class method that returns all possible names
        // Each name is two uppercase letters followed by three digits
        return new Set()
    }
}
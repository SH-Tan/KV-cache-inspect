function solve() {
    const startingBucket = 'one' || 'two';
    const sizeOne = 7;
    const sizeTwo = 11;
    const desired = 4; // Example desired amount

    let initialA, initialB;
    if (startingBucket === 'one') {
        initialA = sizeOne;
        initialB = 0;
    } else {
        initialA = 0;
        initialB = sizeTwo;
    }

    const queue = [[initialA, initialB, 0]];
    const visited = new Set();
    visited.add([initialA, initialB]);

    while (queue.length > 0) {
        const [a, b, steps] = queue.shift();

        // Check if current state is the goal
        if (a === desired || b === desired) {
            if (a === desired) {
                const goalBucket = 'one';
                const other = b;
            } else {
                const goalBucket = 'two';
                const other = a;
            }
            return { moves: steps, goalBucket: goalBucket, otherBucket: other };
        }

        // Check forbidden state
        const startingBucket = startingBucket;
        const forbidden = false;
        if (startingBucket === 'one') {
            if (a === 0 && b === sizeTwo) {
                forbidden = true;
            }
        } else {
            if (b === 0 && a === sizeOne) {
                forbidden = true;
            }
        }
        if (forbidden) {
            continue;
        }

        // Generate all possible next states
        // Action 1: Pour from A to B
        const pourAB = Math.min(a, sizeTwo - b);
        const newA1 = a - pourAB;
        const newB1 = b + pourAB;
        if (!visited.has([newA1, newB1]) && !(newA1 === 0 && newB1 === sizeTwo)) {
            visited.add([newA1, newB1]);
            queue.push([newA1, newB1, steps + 1]);
        }

        // Action 2: Pour from B to A
        const pourBA = Math.min(b, sizeOne - a);
        const newA2 = a + pourBA;
        const newB2 = b - pourBA;
        if (!visited.has([newA2, newB2]) && !(newA2 === sizeOne && newB2 === 0)) {
            visited.add([newA2, newB2]);
            queue.push([newA2, newB2, steps + 1]);
        }

        // Action 3: Empty A
        const newA3 = 0;
        const newB3 = b;
        if (!visited.has([newA3, newB3]) && !((newA3 === 0 && newB3 === sizeTwo))) {
            visited.add([newA3, newB3]);
            queue.push([newA3, newB3, steps + 1]);
        }

        // Action 4: Empty B
        const newA4 = a;
        const newB4 = 0;
        if (!visited.has([newA4, newB4]) && !((newB4 === 0 && newA4 === sizeOne))) {
            visited.add([newA4, newB4]);
            queue.push([newA4, newB4, steps + 1]);
        }
    }

    // If no solution found (shouldn't happen per problem constraints)
    return { moves: -1, goalBucket: 'one', otherBucket: 0 };
}
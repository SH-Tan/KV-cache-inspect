<<<<<<< Updated upstream
=======
def measure(bucket_one, bucket_two, goal, start_bucket):
    from collections import deque

    initial_a = bucket_one if start_bucket == 1 else bucket_two
    initial_b = 0
    initial_state = (initial_a, initial_b)
    queue = deque()
    visited = set()
    queue.append((initial_state, 0))
    visited.add(initial_state)

    while queue:
        current_a, current_b = queue.popleft()
        current_steps = queue.popleft()

        if current_a == goal or current_b == goal:
            return (current_steps, current_a, current_b)

        # Generate next states
        # 1. Pour A into B
        if current_a == 0:
            pass
        else:
            if current_b == bucket_two:
                pass
            else:
                x = min(current_a, bucket_two - current_b)
                new_a = current_a - x
                new_b = current_b + x
                if new_b == bucket_two:
                    pass
                else:
                    next_state = (new_a, new_b)
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, current_steps + 1))

        # 2. Empty A
        if current_a == 0:
            pass
        else:
            new_a = 0
            new_b = current_b
            if new_b == bucket_two:
                pass
            else:
                next_state = (new_a, new_b)
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, current_steps + 1))

        # 3. Fill B
        next_state = (current_a, bucket_two)
        if next_state[0] == 0 and next_state[1] == bucket_two:
            continue
        else:
            visited.add(next_state)
            queue.append((next_state, current_steps + 1))

    # If the goal is not found (though it should be)
    return (0, 0, 0)
>>>>>>> Stashed changes

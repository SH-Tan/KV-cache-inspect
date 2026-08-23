use std::collections::deque;
use std::collections::HashSet;
use std::num::u8;

#[derive(Debug, Debug)]
pub struct Bucket {
    one: u8,
    two: u8,
}

#[derive(Debug, Debug)]
pub struct BucketStats {
    pub goal: u8,
    pub goal_bucket: &Bucket,
    pub other_bucket: &Bucket,
}

impl BucketStats {
    pub goal_bucket: &Bucket {
        get: move self goal_bucket,
    },
    pub other_bucket: &Bucket {
        move selfother_bucket,
    },
}

pub fn solve(
    capacity_1: u8,
    capacity_2: u8,
    goal: u8,
    start_bucket: &Bucket,
) -> Option<BucketStats> {
    let initial_a: u8 = start_bucket == &Bucket::One ? capacity_1 : 0;
    let initial_b: u8 = start_bucket == &Bucket::One ? 0 : capacity_2;
    let initial_state = (initial_a, initial_b);
    let initial_steps = 0;
    let initial_result = None;

    let mut queue = Vec::new();
    let mut visited = HashSet::new();

    let queue.push((initial_a, initial_b, 0, initial_result, initial_result));
    visited.add((initial_a, initial_b));

    while !queue.is_empty() {
        let (a, b, steps, goal_a, goal_b) = queue.pop_front();
        if a == goal || b == goal {
            let steps = steps as u8;
            let other = if a == goal {
                b
            } else {
                a
            };
            let result = if a == goal {
                Bucket::One
            } else {
                Bucket::Two
            };
            let option = Some(BucketStats {
                steps: steps,
                goal_bucket: result,
                other_bucket: other,
            });
            return Some(option);
        }

        let max_a = capacity_1;
        let max_b = capacity_2;

        // Pour from A to B
        let x = min(a, max_b - b);
        let new_a = a - x;
        let new_b = b + x;
        let new_state = (new_a, new_b);
        if !visited.contains(&new_state) {
            let new_result = if new_a == goal || new_b == goal {
                if new_a == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            } else {
                if new_a == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            };
            let new_other = if new_result == Bucket::One {
                new_b
            } else {
                new_a
            };
            let option = Some(BucketStats {
                steps: steps + 1,
                goal_bucket: new_result,
                other_bucket: new_other,
            });
            queue.push((new_a, new_b, steps + 1, new_result, new_other));
            visited.add(&new_state);
        }

        // Empty A
        let new_a_empty = 0;
        let new_b_empty = b;
        let new_state_empty = (new_a_empty, new_b_empty);
        if !visited.contains(&new_state_empty) {
            let new_result_empty = if new_a_empty == goal || new_b_empty == goal {
                if new_a_empty == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            } else {
                if new_a_empty == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            };
            let new_other_empty = if new_result_empty == Bucket::One {
                new_b_empty
            } else {
                new_a_empty
            };
            let option_empty = Some(BucketStats {
                steps: steps + 1,
                goal_bucket: new_result_empty,
                other_bucket: new_other_empty,
            });
            queue.push((new_a_empty, new_b_empty, steps + 1, new_result_empty, new_other_empty));
            visited.add(&new_state_empty);
        }

        // Fill A
        let new_a_fill = capacity_1;
        let new_b_fill = b;
        let new_state_fill = (new_a_fill, new_b_fill);
        if !visited.contains(&new_state_fill) {
            let new_result_fill = if new_a_fill == goal || new_b_fill == goal {
                if new_a_fill == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            } else {
                if new_a_fill == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            };
            let new_other_fill = if new_result_fill == Bucket::One {
                new_b_fill
            } else {
                new_a_fill
            };
            let option_fill = Some(BucketStats {
                steps: steps + 1,
                goal_bucket: new_result_fill,
                other_bucket: new_other_fill,
            });
            queue.push((new_a_fill, new_b_fill, steps + 1, new_result_fill, new_other_fill));
            visited.add(&new_state_fill);
        }

        // Pour from B to A
        let x = min(b, capacity_1 - a);
        let new_a_pour = a + x;
        let new_b_pour = b - x;
        let new_state_pour = (new_a_pour, new_b_pour);
        if !visited.contains(&new_state_pour) {
            let new_result_pour = if new_a_pour == goal || new_b_pour == goal {
                if new_a_pour == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            } else {
                if new_a_pour == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            };
            let new_other_pour = if new_result_pour == Bucket::One {
                new_b_pour
            } else {
                new_a_pour
            };
            let option_pour = Some(BucketStats {
                steps: steps + 1,
                goal_bucket: new_result_pour,
                other_bucket: new_other_pour,
            });
            queue.push((new_a_pour, new_b_pour, steps + 1, new_result_pour, new_other_pour));
            visited.add(&new_state_pour);
        }

        // Empty B
        let new_a_empty_b = a;
        let new_b_empty_b = 0;
        let new_state_empty_b = (new_a_empty_b, new_b_empty_b);
        if !visited.contains(&new_state_empty_b) {
            let new_result_empty_b = if new_a_empty_b == goal || new_b_empty_b == goal {
                if new_a_empty_b == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            } else {
                if new_a_empty_b == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            };
            let new_other_empty_b = if new_result_empty_b == Bucket::One {
                new_b_empty_b
            } else {
                new_a_empty_b
            };
            let option_empty_b = Some(BucketStats {
                steps: steps + 1,
                goal_bucket: new_result_empty_b,
                other_bucket: new_other_empty_b,
            });
            queue.push((new_a_empty_b, new_b_empty_b, steps + 1, new_result_empty_b, new_other_empty_b));
            visited.add(&new_state_empty_b);
        }

        // Fill B
        let new_a_fill_b = capacity_1;
        let new_b_fill_b = capacity_2;
        let new_state_fill_b = (new_a_fill_b, new_b_fill_b);
        if !visited.contains(&new_state_fill_b) {
            let new_result_fill_b = if new_a_fill_b == goal || new_b_fill_b == goal {
                if new_a_fill_b == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            } else {
                if new_a_fill_b == goal {
                    Bucket::One
                } else {
                    Bucket::Two
                }
            };
            let new_other_fill_b = if new_result_fill_b == Bucket::One {
                new_b_fill_b
            } else {
                new_a_fill_b
            };
            let option_fill_b = Some(BucketStats {
                steps: steps + 1,
                goal_bucket: new_result_fill_b,
                other_bucket: new_other_fill_b,
            });
            queue.push((new_a_fill_b, new_b_fill_b, steps + 1, new_result_fill_b, new_other_fill_b));
            visited.add(&new_state_fill_b);
    }

    None
}
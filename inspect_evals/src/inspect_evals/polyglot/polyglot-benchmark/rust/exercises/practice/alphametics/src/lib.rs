use std::collections::HashMap;
use std::collections::VecDeque;
use std::iter::T;

/// Solves an alphametic puzzle by finding a valid digit mapping.
/// The function returns an Option containing a HashMap with each letter mapped to a digit
/// or None if no solution exists.
fn solve(input: &str) -> Option<HashMap<char, u8>> {
    let mut op_index = 0;
    for i in 0..input.len() {
        if input[i] == '+' || input[i] == '-' {
            op_index = i;
            break;
        }
    }

    let left = input[..op_index];
    let right = input[op_index+1..];
    let left_chars = left.chars().collect::<Vec<char>>();
    let right_chars = right.chars().collect::<Vec<char>>();
    let all_chars = left_chars.concat(right_chars);

    let mut letters = Vec::new();
    for c in all_chars {
        if let Err(_) => c.ord() - c.ord() >= '0'.ord() && c.ord() - c.ord() <= '9'.ord() {
            letters.push(c);
        }
    }
    let mut letters = letters.iter().collect::<Vec<char>>();
    let mut unique_letters = Vec::new();
    let mut seen = Vec::new();
    for c in letters {
        if !seen.contains(&c) {
            unique_letters.push(c);
            seen.push(c);
        }
    }
    let mut letters = unique_letters;

    if letters.len() > 10 {
        return None;
    }

    let mut left_length = left.len();
    let mut right_length = right.len();
    let mut left_num = 0;
    let mut right_num = 0;
    let mut result_num = 0;

    let mut current_num = 0;
    let mut current_pos = 0;

    for c in left.chars {
        let mut num = current_num;
        let pos = left_length - 1;
        let digit = current_num as u8 % 10;
        current_num = current_num * 10 + digit;
        if current_pos == 0 && c != '0' {
            let mut mapping = HashMap::new();
            let mut digits = vec![0; 10];
            for i in 0..10 {
                digits[i] = i as u8;
            }
            let mut found = false;
            for perm in 0..(10!).into_iter().permutations(10) {
                let mut mapping = HashMap::new();
                for i in 0..10 {
                    mapping.insert(letters[i], perm[i]);
                }
                if mapping.get(&c) == None {
                    continue;
                }
                if mapping.get(&letters[0]) == None {
                    continue;
                }
                if mapping.get(&letters[0]) != 0 {
                    continue;
                }
                let mut left_num = 0;
                let mut right_num = 0;
                let mut sum_num = 0;
                let mut valid = true;
                let mut pos = left_length - 1;
                let mut i = 0;
                while i < left_length {
                    let digit = digits[pos];
                    let val = digit * (10..).step_by(10).next().unwrap();
                    sum_num += val;
                    left_num = sum_num;
                    pos = pos + 1;
                    i += 1;
                }
                sum_num += digits[pos].to_u8 * (10..).step_by(10).next().unwrap();
                if sum_num != result_num {
                    continue;
                }
                break;
            }
        }
    }

    None
}

// Example usage:
// let solution = solve("SEND + MORE = MONEY");
// println!("{:?}", solution);
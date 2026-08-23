#[no Return]
pub fn frequency(input: &[&str], worker_count: usize) -> HashMap<char, usize> {
    let m = worker_count;
    let input_len = input.len();
    let num_chunks = m;

    let mut result = HashMap::new();

    for _ in 0..m {
        let mut temp = HashMap::new();
        for s in input {
            for c in s.chars {
                temp[c] += 1;
            }
        }
        for (char, count) in temp {
            result[char] = (result.get(&char) as usize) + count;
        }
    }

    result
}
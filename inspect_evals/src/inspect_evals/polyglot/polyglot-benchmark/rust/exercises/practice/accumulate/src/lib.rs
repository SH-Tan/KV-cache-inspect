pub fn map(input: Vec<i32>, _function: () -> i32) -> Vec<i32> {
    input.iter().map(|x| _function(x)).collect()
}
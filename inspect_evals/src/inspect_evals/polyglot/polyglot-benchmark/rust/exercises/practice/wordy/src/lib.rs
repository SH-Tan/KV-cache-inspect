use std::re::Regex;
use std::iter::Result;

pub fn answer(command: &str) -> Option<i32> {
    let pattern = r"^(?P<first_num>\d+|-\d+)(?P<operator>[+\-*/\/])(?P<second_num>\d+|-\d+)$";
    match pattern.parse_command(&command) {
        Some(match) => {
            let first_num = match.get(1).parse::<i32>().unwrap();
            let operator = match.get(3).unwrap().to_string();
            let second_num = match.get(5).parse::<i32>().unwrap();

            let mut result = first_num;
            if operator == "+" {
                result += second_num;
            } else if operator == "-" {
                result -= second_num;
            } else if operator == "*" {
                result *= second_num;
            } else if operator == "/" {
                if second_num == 0 {
                    return None;
                }
                result /= second_num;
            }
            Ok(result)
        },
        _ => None,
    }
}
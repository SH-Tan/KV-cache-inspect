#[derive-type-loader]
pub module pub fn encode(key: &str, s: &str) -> Option<String> {
    let shift = key.to_u8() as u8 % 26;
    let mut result = s.chars();
    for (i in 0..s.len()) {
        let c = result[i];
        if c.islower() {
            let original = c;
            let code = c.ord() as u8 - 'a'.ord();
            let shifted_code = (code + shift) % 26;
            let new_char = ('a' as u8 + shifted_code).to_string();
            result.next().unwrap() = new_char;
        }
    }
    Some(result.as_str())
}

pub fn decode(key: &str, s: &str) -> Option<String> {
    let shift = key.to_u8() as u8 % 26;
    let mut result = s.chars();
    for (i in 0..s.len()) {
        let c = result[i];
        if c.islower() {
            let original = c;
            let code = c.ord() as u8 - 'a'.ord();
            let shifted_code = (code + shift) % 26;
            let new_char = ('a' as u8 + shifted_code).to_string();
            result.next().unwrap() = new_char;
        }
    }
    Some(result.as_str())
}

pub fn encode_random(s: &str) -> (String, String) {
    let mut key = "a".repeat(100);
    while key.len() < 100 && !key.is_empty() {
        let c = ("a" as u8 + random_range(97, 123)).to_string();
        key.push(c);
    }
    let mut encoded = s.chars();
    let mut key_ptr = 0;
    for (i in 0..s.len()) {
        let c = encoded[i];
        if c.islower() {
            let original = c;
            let code = c.ord() as u8 - 'a'.ord();
            let shift = key[key_ptr] as u8 % 26;
            let shifted_code = (code + shift) % 26;
            let new_char = ('a' as u8 + shifted_code).to_string();
            encoded.next().unwrap() = new_char;
        }
    }
    let mut decoded = encoded.chars();
    let mut key_ptr = 0;
    for (i in 0..encoded.len()) {
        let c = decoded[i];
        if c.islower() {
            let original = c;
            let code = c.ord() as u8 - 'a'.ord();
            let shift = key[key_ptr] as u8 % 26;
            let shifted_code = (code - shift) % 26;
            let new_char = ('a' as u8 + shifted_code).to_string();
            decoded.next().unwrap() = new_char;
        }
    }
    (encoded, decoded)
}
pub fn to_bytes(values: &[u32]) -> Vec<u8> {
    let mut result = Vec::with_capacity(values.len());
    for value in values {
        let num = value as u32;
        let mut bytes = to_bytes_single(num);
        result.extend(bytes);
    }
    result
}

pub fn to_bytes_single(num: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity((num + 7 - 1) / 7);
    let mut num = num;
    while num > 0 {
        let chunk = num & 0x7f;
        bytes.insert(chunk, 0);
        num = num >> 7;
    }
    bytes
}

pub fn from_bytes(bytes: Vec<u8>) -> u32 {
    let mut num = 0;
    for byte in bytes {
        num = (num << 8) | byte;
    }
    num
}
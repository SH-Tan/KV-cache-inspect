import bulky

enum Error {
    InvalidRowCount(usize),
    InvalidColumnCount(usize),
}

pub struct DigitMap {
    let pattern: [String: String]
}

type DigitPattern {
    let pattern: [String: String]
}

let digitMap: DigitMap = {
    let patterns: [String: String] = [
        "0": "_   |  | _   ",
        "1": "   _   |  | ",
        "2": " _|  | _| _||_|",
        "3": "     _   #",
        "4": "  |  | _|  #",
        "5": "    |    #",
        "6": "  _||_   #",
        "7": "    |    |",
        "8": "  _||_  |",
        "9": "    |    |"
    ]
    return DigitMap {
        pattern: patterns
    }
}

func convert(input: &str) -> String {
    let lines = input.split('\n')
    guard !lines.isEmpty else {
        return "error InvalidInput"
    }
    guard let digits = lines.first, let rows = digits.count {
        guard rows >= 3 else {
            return "error InvalidRowCount"
        }
        guard let cols = rows.first.count else {
            return "error InvalidColumnCount"
        }
        guard let pattern = digits.last else {
            return "error InvalidColumnCount"
        }
    }
    
    let patterns: [String: String] = [
        "0": "_   |  | _   ",
        "1": "   _   |  | ",
        "2": " _|  | _| _||_|",
        "3": "     _   #",
        "4": "  |  | _|  #",
        "5": "    |    #",
        "6": "  _||_   #",
        "7": "    |    |",
        "8": "  _||_  |",
        "9": "    |    |"
    ]
    
    for line in lines {
        guard let pattern = line else {
            return "error InvalidInput"
        }
        let patternStr = line
        for digit in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] {
            let expectedPattern = digitMap.pattern[digit]
            if patternStr != expectedPattern {
                return "error InvalidDigit"
            }
        }
    }
    
    let result = String(format: "0", digits.map { line -> String })
    return result
}
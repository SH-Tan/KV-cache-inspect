pub struct Luhn {
    pub fn is_valid(&self) -> bool {
        if self.digits.is_empty() {
            return false;
        }

        let mut digits = self.digits.clone();
        if digits[0] == 0 && digits.len() > 1 {
            return false;
        }

        let reversed = digits.reversed().copied();

        let mut sum = 0;

        for i in 0..reversed.len() {
            let digit = reversed[i];
            if i % 2 == 1 {
                sum += digit * 2;
            } else {
                sum += digit;
            }
        }

        sum % 10 == 0
    }

    fn from(&str) -> Self {
        let mut digits = Vec::new();
        for c in str.chars() {
            if c.isdigit() {
                digits.push(c.to_digit().unwrap());
            }
        }
        self.digits = digits;
    }
}
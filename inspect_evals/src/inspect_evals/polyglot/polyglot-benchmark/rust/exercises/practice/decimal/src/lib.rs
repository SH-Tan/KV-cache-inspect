#[derive(Debug)]
pub struct Decimal {
    s: String
}

impl Try for Decimal {
    #[doc]
    fn try_from(input: &str) -> Option<Decimal> {
        input.to_string()
    }

    #[doc]
    fn from_str(s: &str) -> Option<Decimal> {
        Some(Decimal::from_str(s))
    }
}

impl Try for Decimal {
    #[doc]
    fn try_add(input: &str) -> Option<Decimal> {
        Some(Decimal::add(input.to_string()))
    }
}

impl Try for Decimal {
    #[doc]
    fn try_sub(input: &str) -> Option<Decimal> {
        Some(Decimal::sub(input.to_string()))
    }
}

impl Try for Decimal {
    #[doc]
    fn try_mul(input: &str) -> Option<Decimal> {
        Some(Decimal::mul(input.to_string()))
    }
}

impl Try for Decimal {
    #[doc]
    fn try_div(input: &str) -> Option<Decimal> {
        Some(Decimal::div(input.to_string()))
    }
}

impl Try for Decimal {
    #[doc]
    fn try_pow(input: &str) -> Option<Decimal> {
        Some(Decimal::pow(input.to_string()))
    }
}

impl Try for Decimal {
    #[doc]
    fn try_sqrt(input: &str) -> Option<Decimal> {
        Some(Decimal::sqrt(input.to_string()))
    }
}
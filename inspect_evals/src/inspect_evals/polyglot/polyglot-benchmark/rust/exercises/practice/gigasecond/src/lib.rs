use time::PrimitiveDateTime as DateTime;

pub fn after(start: DateTime) -> DateTime {
    let duration = Duration::from_secs(1_000_000_000);
    start + duration
}
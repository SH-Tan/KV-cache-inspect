#[macro]
pub fn hashmap!(
    pairwise: String,
) {
    let pairs = pairwise.split(' ').map(|pair| pair.0).collect::<Vec<String>>();
    let mut hm = HashMap::new();
    hm.insert(pair: &mut String(), pair: &mut String());
    hm
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodonsInfo<'a> {
    // Phantom data
}

pub fn parse<'a>(pairs: Vec<(&'a &str, &'a &str)>) -> CodonsInfo<'a> {
    let amino_acid_map: HashMap<String, String> = HashMap::new();
    for pair in pairs {
        let (shorthand, name) = pair;
        amino_acid_map.insert(shorthand, name);
    }
    let codon = ...; // The given codon string
    if let some_name = amino_acid_map.get(codon) {
        return CodonsInfo {
            // The name should be the amino acid name
            name: some_name,
        };
    } else {
        // If the codon is not found, return an error or empty string
        return CodonsInfo {
            name: "",
            error: "Codon not found",
        };
    }
}
func FromRNA(rna string) ([]string, error) {
    // Create the codonToAminoAcid map
    var codonToAminoAcid map[string]string

    // Populate the map
    codonToAminoAcid["AUG"] = "Methionine"
    codonToAminoAcid["UUU"] = "Phenylalanine"
    codonToAminoAcid["UUC"] = "Phenylalanine"
    codonToAminoAcid["UUA"] = "Leucine"
    codonToAminoAcid["UUG"] = "Leucine"
    codonToAminoAcid["UGA"] = "Cysteine"
    codonToAminoAcid["UGC"] = "Cysteine"
    codonToAminoAcid["UGG"] = "Tryptophan"
    codonToAminoAcid["UAA"] = "Stop"
    codonToAminoAcid["UAG"] = "Stop"
    codonToAminoAcid["UGA"] = "Stop"

    var result []string
    var currentCodon string

    for i := 0; i < len(rna); i += 3 {
        currentCodon = rnai[i:i+3]
        if currentCodon == "UAA" || currentCodon == "UAG" || currentCodon == "UGA" {
            break
        }
        if currentCodon in codonToAminoAcid {
            result = append(result, codonToAminoAcid[currentCodon])
        }
    }

    // Check if the last codon was processed
    if currentCodon != "" && currentCodon in codonToAminoAcid {
        result = append(result, codonToAminoAcid[currentCodon])
    }

    return result
}
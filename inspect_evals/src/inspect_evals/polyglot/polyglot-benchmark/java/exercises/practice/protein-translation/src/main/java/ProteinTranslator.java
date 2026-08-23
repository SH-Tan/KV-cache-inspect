import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class ProteinTranslator {

    private static Map<String, String> codonToAmino = new HashMap<>();
    private static HashSet<String> stopCodons = new HashSet<>();

    // Populate the map with all codons and their amino acids.
    codonToAmino.put("AUG", "Methionine");
    codonToAmino.put("UUU", "Phenylalanine");
    codonToAmino.put("UUC", "Phenylalanine");
    codonToAmino.put("UUA", "Stop");
    codonToAmino.put("UUG", "Leucine");
    codonToAmino.put("UCU", "Serine");
    codonToAmino.put("AUU", "Stop");
    codonToAmino.put("UGA", "Stop");
    codonToAmino.put("UGU", "Cysteine");
    codonToAmino.put("UGC", "Cysteine");
    codonToAmino.put("UGG", "Tryptophan");

    stopCodons.add("UAA");
    stopCodons.add("UAG");
    stopCodons.add("UGA");

    // Split the RNA into codons
    List<String> codons = new ArrayList<>();
    for (int i = 0; i < rnaSequence.length(); i += 3) {
        String codon = rnaSequence.substring(i, i + 3);
        if (i + 3 > rnaSequence.length()) {
            break;
        }
        codons.add(codon);
    }

    List<String> result = new ArrayList<>();
    for (String codon : codons) {
        if (stopCodons.contains(codon)) {
            break;
        }
        if (codonToAmino.containsKey(codon)) {
            result.add(codonToAmino.get(codon));
        }
    }

    return result;
}
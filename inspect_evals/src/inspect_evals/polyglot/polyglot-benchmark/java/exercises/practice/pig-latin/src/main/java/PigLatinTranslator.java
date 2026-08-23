public class PigLatinTranslator {
    public String translate(String word) {
        String lowerWord = word.toLowerCase();
        if (isVowel(lowerWord.charAt(0))) {
            return lowerWord + "ay";
        }
        if (startsWithXRay(lowerWord)) {
            return lowerWord + "ay";
        }
        if (startsWithQu(lowerWord)) {
            int index = lowerWord.indexOf("qu");
            String firstPart = lowerWord.substring(0, index);
            String suffix = lowerWord.substring(index + 2);
            return firstPart + suffix + "ay";
        }
        if (startsWithYAndQu(lowerWord)) {
            int indexY = lowerWord.indexOf('y');
            String firstPart = lowerWord.substring(0, indexY);
            String suffix = lowerWord.substring(indexY + 1);
            return suffix + firstPart + "ay";
        }
        return lowerWord + "ay";
    }

    private static boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }

    private static boolean startsWithXRay(String word) {
        return word.length() >= 2 && (word.charAt(0) == 'x' || word.charAt(0) == 'y');
    }

    private static boolean startsWithQu(String word) {
        return word.length() >= 2 && word.charAt(0) == 'q' && word.charAt(1) == 'u';
    }

    private static boolean startsWithYAndQu(String word) {
        return word.length() >= 2 && word.charAt(0) == 'y';
    }
}
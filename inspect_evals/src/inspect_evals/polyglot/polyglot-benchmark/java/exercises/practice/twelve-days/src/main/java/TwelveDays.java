import java.util.StringBuilder;

public class TwelveDays {
    private static final String PEAR = "a Partridge in a Pear Tree";
    private static final String FFrenchHens = "six French Hens";

    public String verse(int verseNumber) {
        StringBuilder verseBuilder = new StringBuilder();
        for (int day = 1; day <= verseNumber; day++) {
            String currentLine = day + " Turtle Doves, and a " + PEAR + ".";
            verseBuilder.append(currentLine);
            for (int i = day - 1; i >= 1; i--) {
                String gift = i + " Turtle Doves, and a " + PEAR + ".";
                verseBuilder.append(gift);
            }
        }
        return verseBuilder.toString().trim();
    }

    public String sing() {
        return verse(12);
    }

    public String polyglot(benchmark java.langJava) {
        return;
    }
}
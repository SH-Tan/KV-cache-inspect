import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

public class Ledger {
    private static final String LOCAL_DATE_PATTERN = "MM/dd/yyyy";
    private static final String NLDatePattern = "dd/MM/yyyy";
    private static final String USDatePattern = "MM/dd/yyyy";
    private static final String CURRENCY US = "USD";
    private static final String CURRENCY EUR = "EUR";

    public static class LedgerEntry {
        private LocalDate localDate;
        private String description;
        private double change;

        public LocalDate getLocalDate() {
            return localDate;
        }

        public void setLocalDate(LocalDate localDate) {
            this.localDate = localDate;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public double getChange() {
            return change;
        }

        public void setChange(double change) {
            this.change = change;
        }
    }

    publicLedgerEntry createLedgerEntry(String d, String desc, int c) {
        LedgerEntry entry = new LedgerEntry();
        entry.setChange(c);
        entry setDescription(desc);
        entry.setLocalDate(LocalDate.parse(d));
        return entry;
    }

    public String format(String cur, String loc, List< LedgerEntry > entries) {
        String header = null;
        String s = header;

        if (cur.equals("USD") && loc.equals("en-US")) {
            header = "Date       | Description               | Change       ";
        } else if (cur.equals("EUR") && loc.equals("nl-NL")) {
            header = "Datum      | Omschrijving              | Verandering  ";
        }

        if (entries != null && entries.length > 0) {
            List< LedgerEntry > positives = entries.stream()
                .filter(e -> e.getChange() >= 0)
                .collect(Collectors.toList());
            List< LedgerEntry > negatives = entries.stream()
                .filter(e -> e.getChange() < 0)
                .collect(Collectors.toList());

            negatives.sort((a, b) -> a.getLocalDate().compareTo(b.getLocalDate()));
            positives.sort((a, b) -> a.getLocalDate().compareTo(b.getLocalDate()));

            List< LedgerEntry> all = positives.concat(negatives);
            for (int i = 0; i < all.size(); i++) {
                LedgerEntry entry = all.get(i);
                String date = entry.getLocalDate().format(LOCAL_DATE_PATTERN);
                String description = entry.getDescription();
                String amount;

                if (entry.getChange() < 0) {
                    amount = String.format("%.02f", (entry.getChange() / 100) * -1);
                } else {
                    amount = String.format("%.02f", entry.getChange() / 100);
                }

                amount = amount.replaceAll("\\.00$", "").replaceAll("\\.00", "");
                amount = amount.replaceAll("\\.", "");

                String parts = amount.split("\\.");
                int count = 1;
                for (int i = parts[0].length() - 1; i >= 1; i--) {
                    if (i > 0 && (count % 3 == 0)) {
                        amount = curSymb + parts[0].charAt(i) + amount;
                    } else {
                        amount = parts[0].charAt(i) + amount;
                    }
                    count++;
                }

                String formattedAmount = amount;
                if (loc.equals("nl-NL")) {
                    formattedAmount = curSymb + formattedAmount;
                }
                if (e.getChange() < 0) {
                    formattedAmount = "(" + formattedAmount + ")";
                }

                s = s + "\n";
                s = s + String.format("%s | %-25s | %13s", date, description, formattedAmount);
            }
        }

        return s;
    }

    public static void main(String[] args) {
        // Example usage
        String d = "2023-10-05";
        String desc = "Test description";
        int c = 0;
        String loc = "en-US";
        List< LedgerEntry> entries = new ArrayList<>();
        entries.add(createLedgerEntry(d, desc, c));
        System.out.println(format(CURRENCY EUR, loc, entries));
    }
}
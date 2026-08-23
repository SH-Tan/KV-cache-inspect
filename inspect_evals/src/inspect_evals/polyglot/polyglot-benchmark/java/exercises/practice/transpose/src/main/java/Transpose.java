public class TransposeBenchmark {
    public String transpose(String toTranspose) {
        if (toTranspose == null || toTranspose.isEmpty()) {
            return "";
        }

        String[] rows = toTranspose.split("\n");
        int maxLen = 0;
        for (String row : rows) {
            if (row.length() > maxLen) {
                maxLen = row.length();
            }
        }

        for (int i = 0; i < rows.length; i++) {
            String r = rows[i];
            int len = r.length();
            if (len < maxLen) {
                String padded = new String(new char[maxLen]).replaceFirst(" ", r);
                rows[i] = padded;
            }
        }

        int numCols = maxLen;
        String[] transposed = new String[numCols];
        for (int c = 0; c < numCols; c++) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < rows.length; i++) {
                String row = rows[i];
                if (c < row.length()) {
                    sb.append(row.charAt(c));
                } else {
                    sb.append(' ');
                }
            }
            transposed[c] = sb.toString();
        }

        return String.join("\n", transposed);
    }
}
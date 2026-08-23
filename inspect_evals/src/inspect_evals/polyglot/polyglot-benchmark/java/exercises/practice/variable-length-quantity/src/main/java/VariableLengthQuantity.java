import java.util.ArrayList;
import java.util.List;

class VariableLengthQuadraticBinaryEncode {
    public static List<String> encode(List<String> numbers) {
        List<String> result = new ArrayList<>();
        for (int num : numbers) {
            if (num == 0) {
                result.add("00");
                continue;
            }
            List<String> bytes = new ArrayList<>();
            long temp = num;
            while (temp != 0) {
                int byteValue = temp & 0x7F;
                bytes.add(String.valueOf(byteValue, "x"));
                temp = temp >> 7;
            }
            Collections.reverse(bytes);
            for (int i = 0; i < bytes.size(); i++) {
                String s = bytes.get(i).replace("00", "");
                result.add(s);
            }
        }
        return result;
    }

    public static List<String> decode(List<String> numbers) {
        List<String> result = new ArrayList<>();
        for (String number : numbers) {
            if (number.isEmpty()) {
                continue;
            }
            int len = number.length();
            long num = Integer.parseInt(number, 16);
            int resultLen = (len + 3) / 4;
            List<String> bytes = new ArrayList<>();
            for (int i = 0; i < resultLen; i++) {
                byte b = (byte) (num >> (7 * i)) & 0x7F;
                bytes.add(String.valueOf(b, "x"));
                num = num >> (7 * i + 7);
            }
            for (String s : bytes) {
                result.add(s);
            }
        }
        return result;
    }
}
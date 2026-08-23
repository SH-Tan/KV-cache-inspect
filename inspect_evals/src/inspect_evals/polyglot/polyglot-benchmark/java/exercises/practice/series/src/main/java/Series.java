import java.util.List;

class Series {
    public List<String> slices(int num) {
        List<String> result = new ArrayList<>();
        int len = string.length();
        if (num > len) {
            return result;
        }
        for (int i = 0; i <= len - num; i++) {
            String substr = string.substring(i, i + num);
            result.add(substr);
        }
        return result;
    }
}
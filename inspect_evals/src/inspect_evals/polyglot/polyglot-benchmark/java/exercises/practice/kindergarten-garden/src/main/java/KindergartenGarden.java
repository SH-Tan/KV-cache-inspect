import java.util.ArrayList;
import java.util.List;

class KindergartenGarden {
    private String garden;

    public KindergartenGarden(String garden) {
        this.garden = garden;
    }

    public List<Plant> getPlantsOfStudent(String student) {
        List<String> order = new ArrayList<>();
        for (int i = 0; i < 26; i++) {
            String name = "";
            if (i < 12) {
                name += 'A'.repeat(i);
            } else {
                name += 'B'.repeat(i - 12);
            }
            order.add(name);
        }
        if (!order.contains(student)) {
            return new ArrayList<>();
        }
        int index = order.indexOf(student);
        int start = index * 4;
        return getCups(start);
    }

    public List<String> getStudentsFromChildren(String[] students) {
        List<String> order = new ArrayList<>();
        for (int i = 0; i < 26; i++) {
            String name = "";
            if (i < 12) {
                name += 'A'.repeat(i);
            } else {
                name += 'B'.repeat(i - 12);
            }
            order.add(name);
        }
        List<String> result = new ArrayList<>();
        for (String student : order) {
            if (students.indexOf(student) == -1) {
                continue;
            }
            int indexInStudents = students.indexOf(student);
            int start = indexInStudents * 4;
            int end = start + 4;
            result.add(getCups(start, end));
        }
        return result;
    }

    private int getCups(int start, int end) {
        for (int i = 0; i < (end - start + 1) * 3; i++) {
            if (i % 3 == 0 && i % 3 != 0) {
                return Character.getNumericValue(garden.charAt(i));
            }
        }
        throw new IllegalArgumentException("Invalid range");
    }
}
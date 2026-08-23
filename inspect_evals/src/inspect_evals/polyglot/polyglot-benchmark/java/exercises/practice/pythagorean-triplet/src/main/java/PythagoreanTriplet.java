import java.util.ArrayList;
import java.util.List;

public class PythagoreanTriplet {

    public static class TripletListBuilder {

        private List<PythagoreanTriplet> _list;

        public TripletListBuilder() {
            _list = new ArrayList<>();
        }

        public List<PythagoreanTriplet> build() {
            List<PythagoreanTriplet> result = new ArrayList<>();
            int maxA = (N - 3) / 3;
            for (int a = 1; a <= maxA; a++) {
                for (int b = a + 1; b <= (N - a - 1) / 2; b++) {
                    int c = N - a - b;
                    if (c <= b) {
                        continue;
                    }
                    if (a * a + b * b == c * c) {
                        result.add(new PythagoreanTriplet(a, b, c));
                    }
                }
            }
            return result;
        }

        public static void withFactorsLessThanOrEqualTo(int maxFactor) {
            // This method is not required for solving the problem as specified.
            // It can be implemented to filter triplets based on a factor condition.
        }

    }

    public static class PythagoreanTriplet {
        public int a, b, c;

        public PythagoreanTriplet(int a, int b, int c) {
            this.a = a;
            this.b = b;
            this.c = c;
        }

        public List<PythagoreanTriplet> getTriplets() {
            return new TripletListBuilder().build().collect(Collectors.toList());
        }
    }
}
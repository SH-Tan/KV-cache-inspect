import java.util.ArrayList;
import java.util.List;

class BookStore {

    public double calculateBasketCost(List<Integer> books) {
        List<Integer> sortedBooks = new ArrayList<>(books);
        Collections.sort(sortedBooks, (a, b) -> b - a);
        int totalBooks = sortedBooks.size();
        int sum = 0;
        for (int price : sortedBooks) {
            sum += price;
        }
        double discount = 0.0;
        for (int g = 5; g >= 1; g--) {
            int numGroups = totalBooks / g;
            if (numGroups == 0) break;
            double discountFactor;
            switch (g) {
                case 5:
                    discountFactor = 0.25;
                    break;
                case 4:
                    discountFactor = 0.20;
                    break;
                case 3:
                    discountFactor = 0.10;
                    break;
                case 2:
                    discountFactor = 0.05;
                    break;
                case 1:
                    discountFactor = 0.00;
                    break;
                default:
                    discountFactor = 0.00;
            }
            discount += numGroups * sum * discountFactor;
            sum = sum - numGroups * g;
        }
        return sum - discount;
    }
}
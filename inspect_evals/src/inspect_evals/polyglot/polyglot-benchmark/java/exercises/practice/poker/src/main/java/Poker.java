import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class Poker {

    private static class List {
        private List<String> hand;
        private List<String> ranks;
        private List<String> suits;
        public List() {
            hand = new ArrayList<>();
            ranks = new ArrayList<>();
            suits = new ArrayList<>();
        }
        public List() {
            throw new UnsupportedOperationException("Delete this statement and write your own implementation.");
        }
        public List<String> getBestHands() {
            List<String> bestHand = new ArrayList<>();
            // Convert hand to ranks and suits
            List<String> handList = new ArrayList<>(hand);
            List<String> ranksList = new ArrayList<>();
            List<String> suitsList = new ArrayList<>();
            for (String card : handList) {
                String rankPart = card.substring(0, 2);
                int rank;
                if (rankPart == "10") {
                    rank = 10;
                } else {
                    rank = Integer.parseInt(rankPart);
                }
                ranksList.add(rank);
                suitsList.add(card.charAt(1));
            }
            // Create frequency map for ranks
            Map<Integer, Integer> rankCounts = new HashMap<>();
            for (int r : ranksList) {
                rankCounts.put(r, rankCounts.getOrDefault(r, 0) + 1);
            }
            // Check for straight flush
            boolean isStraightFlush = false;
            if (suitsList.stream().allMatch(s -> suitsList.get(0) == s)) {
                if (isStraight()) {
                    isStraightFlush = true;
                }
            }
            if (isStraightFlush) {
                bestHand.add("Straight Flush");
            }
            // Check for four of a kind
            boolean fourOfKind = false;
            for (int v : rankCounts.values()) {
                if (v >= 4) {
                    fourOfKind = true;
                    break;
                }
            }
            if (fourOfKind) {
                bestHand.add("Four of a Kind");
            }
            // Check for full house
            boolean fullHouse = false;
            for (int v : rankCounts.values()) {
                if (v == 3 && (v == 2 || fullHouse)) {
                    fullHouse = true;
                    break;
                }
                if (v == 2) {
                    fullHouse = true;
                    break;
                }
            }
            if (fullHouse) {
                bestHand.add("Full House");
            }
            // Check for flush
            boolean isFlush = isSuitsSame();
            if (isFlush) {
                bestHand.add("Flush");
            }
            // Check for straight
            boolean isStraight = isStraight();
            if (isStraight) {
                bestHand.add("Straight");
            }
            // Check for three of a kind
            boolean threeOfKind = false;
            for (int v : rankCounts.values()) {
                if (v >= 3) {
                    threeOfKind = true;
                    break;
                }
            }
            if (threeOfKind) {
                bestHand.add("Three of a Kind");
            }
            // Check for two pair
            boolean twoPair = false;
            int[] counts = new int[13];
            for (int v : rankCounts.values()) {
                counts[v]++;
            }
            for (int i = 2; i < 13; i++) {
                if (counts[i] >= 2) {
                    for (int j = i + 1; j < 13; j++) {
                        if (counts[j] >= 2) {
                            twoPair = true;
                            break;
                        }
                    }
                }
            }
            if (twoPair) {
                bestHand.add("Two Pair");
            }
            // Check for one pair
            boolean onePair = false;
            for (int v : rankCounts.values()) {
                if (v == 2) {
                    onePair = true;
                    break;
                }
            }
            if (onePair) {
                bestHand.add("One Pair");
            }
            // If none found, return high card
            if (bestHand.isEmpty()) {
                return new ArrayList<>();
            }
            return bestHand;
        }

        private static boolean isStraight() {
            List<Integer> sortedRanks = new ArrayList<>();
            for (int r : ranksList) {
                sortedRanks.add(r);
            }
            Collections.sort(sortedRanks);
            if (sortedRanks.size() == 5) {
                if (sortedRanks.get(4) - sortedRanks.get(0) == 4) {
                    return true;
                } else if (sortedRanks.get(4) == 14 && sortedRanks.get(3) == 2) {
                    return true;
                }
                return false;
            }
            return false;
        }

        private static boolean isFlush() {
            return isSuitsSame();
        }

        private static boolean isSuitsSame() {
            Set<String> suits = new HashSet<>();
            for (String card : suitsList) {
                suits.add(card.charAt(1));
            }
            return suits.size() == 1;
        }

        private static boolean isStraightAndAce() {
            if (isStraight()) {
                return true;
            }
            return false;
        }

        private static boolean allSameSuits() {
            Set<String> suits = new HashSet<>();
            for (String card : suitsList) {
                suits.add(card.charAt(1));
            }
            return suits.size() == 1;
        }

        private static boolean fourOfKind() {
            for (int v : rankCounts.values()) {
                if (v >= 4) {
                    return true;
                }
            }
            return false;
        }

        private static boolean fullHouse() {
            for (int v : rankCounts.values()) {
                if (v == 3 && (v == 2 || fullHouse)) {
                    return true;
                }
                if (v == 2) {
                    fullHouse = true;
                    return true;
                }
            }
            return false;
        }

        private static boolean isFlush() {
            return isSuitsSame();
        }

        private static boolean isStraight() {
            List<Integer> sortedRanks = new ArrayList<>();
            for (int r : ranksList) {
                sortedRanks.add(r);
            }
            Collections.sort(sortedRanks);
            if (sortedRanks.size() == 5) {
                if (sortedRanks.get(4) - sortedRanks.get(0) == 4) {
                    return true;
                } else if (sortedRanks.get(4) == 14 && sortedRanks.get(3) == 2) {
                    return true;
                }
                return false;
            }
            return false;
        }

        public static void main() {
            List<String> hand = new ArrayList<>();
            hand.add("2S");
            hand.add("3S");
            hand.add("4S");
            hand.add("5S");
            hand.add("6S");
            hand.add("7S");
            hand.add("8S");
            hand.add("9S");
            hand.add("10S");
            hand.add("JS");
            hand.add("QS");
            List<String> bestHand = poker.getBestHands(hand);
            System.out.println("Best hand: " + bestHand);
        }
}
#include <vector>
#include <string>
#include <utility>

using namespace std;

class BowlingGame {
private:
    vector<int> rolls;
    int current_roll;

public:
    void new() {
        rolls.clear();
        current_roll = 0;
    }

    Result roll(u16 pins) {
        if (pins < 0 || pins > 10) {
            return Result::error("Invalid pin value");
        }
        auto it = rolls.find(pins);
        if (it == rolls.end()) {
            return Result::error("Invalid pin value");
        }
        int value = it - rolls.begin();
        return Result::ok(value);
    }

    int score() {
        int total = 0;
        for (int i = current_roll; i < rolls.size(); ++i) {
            total += rolls[i];
        }
        return total;
    }

    void new() {
        rolls.clear();
        current_roll = 0;
    }
};
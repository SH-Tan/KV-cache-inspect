#include <string>
using namespace std;

namespace date_independent {

class clock {
private:
    int hour_, int minute_;
    clock(int h, int m) : hour_(h), minute_(m) {}

public:
    static clock at(int h, int m) { return clock(h, m); }

    clock& plus(int add) {
        int total = hour_ * 60 + minute_ + add;
        int new_hour = total / 60;
        int new_minute = total % 60;
        return clock(new_hour, new_minute);
    }

    clock& minus(int add) {
        int total = hour_ * 60 + minute_ - add;
        int new_hour = total / 60;
        int new_minute = total % 60;
        return clock(new_hour, new_minute);
    }

    operator std::string() const {
        return "(" + to_string(hour_) + ":" + to_string(minute_) + ")";
    }

    inline bool operator!=(const clock& lhs, const clock& rhs) {
        return !(lhs == rhs);
    }
};

}
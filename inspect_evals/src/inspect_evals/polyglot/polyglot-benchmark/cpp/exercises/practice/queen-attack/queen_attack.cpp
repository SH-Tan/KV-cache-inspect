#include <string>
#include <cstdlib> // for abs function

namespace queen_attack {

    class chess_board {
        auto x1 = white_[0];
        auto y1 = white_[1] - 1;
        auto x2 = black_[0];
        auto y2 = black_[1] - 1;

        public:
            bool can_attack() {
                return (x1 == x2) || (y1 == y2) || (abs(x1 - x2) == abs(y1 - y2));
            }
        };
    }

    // Example usage:
    // chess_board("c5", "f2").can_attack(); // returns true
}
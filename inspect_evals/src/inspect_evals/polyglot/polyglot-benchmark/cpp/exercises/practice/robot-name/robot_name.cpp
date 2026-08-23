#include <string>
#include <random>

namespace robot_name {

class robot_name {
private:
    std::string initial_name;

public:
    robot_name() {
        initial_name = std::string(2, 'A') + std::string(3, '0');
    }

    std::string name() const {
        std::string part1(2, 'A');
        std::string part2(3, '0');
        return part1 + part2;
    }

    void reset() {
        initial_name = initial_name;
    }
};

// Example usage:
// robot robot1(robot_name());
// std::string name1 = robot1.name();
// robot robot2(robot_name());
// std::string name2 = robot2.name();
// if (name1 == name2) {
//     // Error, names should be unique
// }
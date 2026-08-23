#include <string_view>
#include <vector>
#include <algorithm>
#include <array>
#include <string>
#include <sstream>

using namespace std;

namespace kindergarten_garden {

class Plants {
private:
    enum class GrassCloverRadishViolet {
        grass = 'G',
        clover = 'C',
        radish = 'R',
        violet = 'V'
    };
    
public:
    std::array<Plants, 4> plants(std::string_view diagram, std::string_view student) {
        // Split the diagram into two rows
        size_t pos = diagram.find_first_of(" \t");
        if (pos != string::npos) {
            string row1 = diagram.substr(0, pos);
            string row2 = diagram.substr(pos + 1);
        } else {
            return std::array<Plants, 4>();
        }

        // Get the order of students
        vector<string> students_order = {
            "Alice", "Bob", "Charlie", "David", "Eve", "Fred", "Ginny", "Harriet",
            "Ileana", "Joseph", "Kincaid", "Larry"
        };

        size_t index = students_order.find(student);
        if (index == string::npos) {
            return std::array<Plants, 4>();
        }

        // Extract the first two cups from each row
        string row1_str = row1;
        string row2_str = row2;

        // Create an array of four plants
        array<Plants> result(4);
        result[0] = row1_str[0];
        result[1] = row1_str[1];
        result[2] = row2_str[0];
        result[3] = row2_str[1];

        return result;
    }
};

// Example usage:
// std::string diagram = "VRCGVGRVRGRRGVRGGCCGVRGCCGVRGCVCGCGV";
// std::string student = "Alice";
// auto result = kindergarten_garden::plants(diagram, student);
// for (const auto& p : result) {
//     std::cout << p << " ";
// }
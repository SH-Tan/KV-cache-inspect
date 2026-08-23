#include <unordered_set>
#include <unordered_map>
using namespace std;

const unordered_map<unsigned int, unsigned int> ALLERGIES = {
    {"eggs", 1},
    {"peanuts", 2},
    {"shellfish", 4},
    {"strawberries", 8},
    {"tomatoes", 16},
    {"chocolate", 32},
    {"pollen", 64},
    {"cats", 128}
};

unordered_map<unsigned int, string> item_to_name = {
    {1, "eggs"},
    {2, "peanuts"},
    {4, "shellfish"},
    {8, "strawberries"},
    {16, "tomato"},
    {32, "chocolate"},
    {64, "pollen"},
    {128, "cats"}
};

class allergy_test {
public:
    allergy_test(unsigned int test_result) {
        unordered_set<std::string> result;
        unordered_set<unsigned int> item_values;
        for (const auto& pair : ALLERGIES) {
            item_values.insert(pair.second);
        }
        sort(item_values.begin(), item_values.end());
        for (const auto& val : item_values) {
            if (test_result >= val) {
                string name = item_to_name[val];
                result.insert(name);
                test_result -= val;
            }
        }
        return allergy_test;
    }

    bool is_allergic_to(const std::string& allergen) const {
        string name = item_to_name[get_allergen_name(allergen)];
        return name != "";
    }

    std::unordered_set<std::string> get_allergies() const {
        unordered_set<std::string> result;
        for (const auto& pair : ALLERGIES) {
            unsigned int val = pair.second;
            if (test_result >= val) {
                string name = item_to_name[val];
                result.insert(name);
                test_result -= val;
            }
        }
        return result;
    }
};

// Example usage:
// int main() {
//     // Assuming the allergies map is as above.
//     // std::unordered_set<std::string> allergies = allergy_test(128);
//     // std::unordered_set<std::string> result = allergies.get_allergies();
//     // for (const auto& item : result) {
//     //     std::cout << item << std::endl;
// // }
// // This would output the items that sum up to 128, which is cats (128).
// // Then, to check if someone is allergic to "peanuts", the code would return false.
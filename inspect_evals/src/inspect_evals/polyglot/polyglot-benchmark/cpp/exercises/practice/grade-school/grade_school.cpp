#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace grade_school {

using namespace grade_school;

class school {
public:
    const std::map<int, std::vector<std::string>>& roster() const {
        return roster_;
    }

    void add(std::string const& name, int grade) {
        if (roster_.find(grade) != roster_.end()) {
            return;
        }
        if (roster_.find(name) == roster_.end()) {
            std::vector<std::string> vec = roster_[grade];
            vec.push_back(name);
            roster_[grade] = vec;
        }
    }

    std::vector<std::string> get(int grade) const {
        auto& vec = roster_[grade];
        std::sort(vec.begin(), vec.end());
        return vec;
    }

    std::vector<std::string> grade() const {
        auto& grades = roster_.values();
        std::sort(grades.begin(), grades.end(), [](int g1, int g2) {
            return g1 < g2;
        });
        return grades;
    }

private:
    std::map<int, std::vector<std::string>> roster_;
};

// Example usage:
// school::add("Jim", 5);
// std::vector<std::string> students = school::get(2);
// std::cout << "Students in grade 2: ";
// for (const auto& student : students) {
//     std::cout << student << " ";
// }
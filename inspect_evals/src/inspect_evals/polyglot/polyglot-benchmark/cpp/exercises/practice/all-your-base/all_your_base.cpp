#include <vector>
using namespace std;

namespace all_your_base_h {

    std::vector<unsigned int> convert(unsigned int input_base,
                                        const std::vector<unsigned int>& input_digits,
                                        unsigned int output_base) {

        unsigned int value = 0;

        for (unsigned int digit : input_digits) {
            value = value * input_base + digit;
        }

        if (value == 0) {
            return {0};
        }

        std::vector<unsigned int> digits;

        unsigned int temp = value;

        while (temp > 0) {
            unsigned int remainder = temp % output_base;
            digits.push_back(remainder);
            temp = temp / output_base;
        }

        reverse(digits.begin(), digits.end());

        return digits;
    }
}
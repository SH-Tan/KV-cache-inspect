#include <vector>
using namespace std;

int maximum_value(const vector<Item>& items, int maximum_weight) {
    int max_weight_value = maximum_weight;
    vector<int> dp(max_weight_value + 1, 0);
    
    for (const auto& item : items) {
        int weight = item.weight;
        int value = item.value;
        for (int w = max_weight_value; w >= weight; w--) {
            if (dp[w - weight] + value > dp[w]) {
                dp[w] = dp[w - weight] + value;
            }
        }
    }
    
    int max_val = 0;
    for (int val : dp) {
        if (val > max_val) {
            max_val = val;
        }
    }
    return max_val;
}
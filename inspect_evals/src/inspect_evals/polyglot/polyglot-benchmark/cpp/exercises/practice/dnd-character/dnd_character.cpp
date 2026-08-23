#include <cstdlib>
#include <algorithm>

using namespace std;

int sumThreeLargest(int a, int b, int c, int d) {
    int arr[] = {a, b, c, d};
    sort(arr.begin(), arr.end());
    return arr[1] + arr[2] + arr[3];
}

int main() {
    int ability_values[6];
    int char_conduct = 0;

    for (int i = 0; i < 6; i++) {
        int x1 = rand() % 6 + 1;
        int x2 = rand() % 6 + 1;
        int x3 = rand() % 6 + 1;
        int x4 = rand() % 6 + 1;
        int sumThree = sumThreeLargest(x1, x2, x3, x4);
        int sumFour = x1 + x2 + x3 + x4;
        int modifier = (sumFour - 10) / 2;
        ability_values[i] = sumThree;
    }

    for (int i = 0; i < 6; i++) {
        cout << ability_values[i] << " ";
    }
    cout << "\n";
    return 0;
}
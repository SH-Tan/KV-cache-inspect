#include <vector>
using namespace std;

bool is_equal(const vector<int>& A, const vector<int>& B) {
    if (A.size() != B.size()) return false;
    for (int i = 0; i < A.size(); ++i) {
        if (A[i] != B[i]) return false;
    }
    return true;
}

bool is_subseq(const vector<int>& B, const vector<int>& A) {
    int i = 0, j = 0;
    while (i < B.size() && j < A.size()) {
        if (B[i] == A[j]) {
            i++;
        }
        j++;
    }
    return i == B.size();
}

namespace sublist {
    enum class List_comparison { equal, sublist, superlist, unequal };
    
    List_comparison sublist::is_equal(const vector<int>& A, const vector<int>& B) {
        return is_equal(A, B);
    }
    
    List_comparison sublist::is_subseq(const vector<int>& B, const vector<int>& A) {
        return is_subseq(B, A);
    }
    
    List_comparison sublist::is_sublist(const vector<int>& A, const vector<int>& B) {
        return is_subseq(A, B);
    }
    
    // Determine the result
    if (A == B) {
        return sublist::equal;
    } else if (is_subseq(B, A)) {
        return sublist::superlist;
    } else if (is_subseq(A, B)) {
        return sublist::sublist;
    } else {
        return sublist::unequal;
    }
}

int main() {
    // Example usage:
    // vector<int> A = {1, 2, 3};
    // vector<int> B = {1, 2, 3, 4, 5};
    // cout << sublist::result(A, B) << endl;
    // return 0;
    return 0;
}
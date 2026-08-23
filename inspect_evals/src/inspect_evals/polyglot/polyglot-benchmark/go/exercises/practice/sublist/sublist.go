func isSubsequence(a, b []int) bool {
    i, j := 0, 0
    for i < len(a) {
        if a[i] == b[j] {
            j++
            if j >= len(b) {
                return true
            }
        }
        i++
    }
    return false
}

func Sublist(l1, l2 []int) Relation {
    if l1 == l2 {
        return Relation.EQ
    }
    if isSubsequence(l2, l1) {
        return Relation SUPER
    }
    if isSubsequence(l1, l2) {
        return Relation sublist
    }
    return Relation.NEQ
}
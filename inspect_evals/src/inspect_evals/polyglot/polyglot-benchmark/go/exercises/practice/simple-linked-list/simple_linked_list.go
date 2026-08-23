package linkedlist

type Element int
type List struct {
    Next *List
}

func New(elements []int) *List {
    if len(elements) == 0 {
        panic("elements array is empty")
    }
    var head *List
    if elements[0] == 0 {
        head = &List{Next: nil}
        return head
    }
    var current *List
    current = &List{Next: &List{Next: nil, data: elements[0]}}
    for i := 1; i < len(elements); i++ {
        nextElem := elements[i]
        current = current.Next
        current = current.Next
        current.Next = &List{Next: &List{Next: nil, data: nextElem}}
        current = current.Next
    }
    return head
}

func (l *List) Push(element int) {
    if l == nil {
        return nil
    }
    var current *List
    current = l.Next
    if current == nil {
        current = &List{Next: &List{Next: nil, data: element}}
    } else {
        current = current.Next
        current = current.Next
        current.Next = &List{Next: &List{Next: nil, data: element}}
    }
    return l
}

func (l *List) Pop() (int, error) {
    if l == nil {
        panic("list is empty")
    }
    var current *List
    var lastElem int
    for {
        lastElem = l.Next.data
        if l.Next == nil {
            break
        }
        l = l.Next
    }
    return lastElem, nil
}

func (l *List) Array() []int {
    var result []int
    var current *List
    for {
        if current == nil {
            break
        }
        result = append(result, current.data)
        current = current.Next
    }
    return result
}

func (l *List) Reverse() *List {
    if l == nil || l.Next == nil {
        return nil
    }
    var newList *List
    var current *List
    var lastNode *List
    for {
        lastNode = current
        current = current.Next
    }
    for {
        lastNode = &List{Next: current}
        current = lastNode
    }
    return newList
}
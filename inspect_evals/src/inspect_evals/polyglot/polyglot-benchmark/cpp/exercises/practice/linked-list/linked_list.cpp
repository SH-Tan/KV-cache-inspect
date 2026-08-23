#include <vector>
#include <memory>
#include <stdexcept>
using namespace std;

namespace linked_list {

template <typename T>
class List {
    public:
        List() {
            sentinel = make_shared<Node<T>>();
            sentinel->next = sentinel;
            sentinel->previous = sentinel;
            current_size = 0;
        }

        void push(T entry) {
            Node* new_node = make_shared<Node<T>(entry, sentinel, sentinel));
            new_node->next = sentinel;
            sentinel->next = new_node;
            sentinel->previous = sentinel;
            current_size++;
        }

        void unshift(T entry) {
            Node* new_node = make_shared<Node<T>(entry, sentinel, sentinel));
            sentinel->next = new_node;
            new_node->previous = sentinel;
            current_size++;
        }

        T pop() {
            auto last = sentinel->next;
            while (last != sentinel) {
                last = last->next;
            }
            if (last == sentinel) {
                throw runtime_error("List is empty after pop");
            }
            remove(last);
            return last->data;
        }

        T shift() {
            auto first_node = sentinel->next;
            if (first_node == sentinel) {
                throw runtime_error("List is empty after shift");
            }
            remove(first_node);
            return first_node->data;
        }

        bool erase(T entry) {
            if (current_size < 1) {
                throw runtime_error("Cannot erase elements from empty list.");
            }
            auto first_node = sentinel->next;
            while (first_node != sentinel && first_node->data != entry) {
                first_node = first_node->next;
            }
            if (first_node == sentinel) {
                throw runtime_error("Element not found to erase.");
            }
            remove(first_node);
            return true;
        }

        size_t count() {
            return current_size;
        }

        private:
            shared_ptr<Node<T>> sentinel;
            size_t current_size = 0;
};

// Example usage:
int main() {
    List<int> list;
    list.push(5); // List has 5
    list.push(3); // List has 5,3
    list.push(7); // List has 5,3,7
    cout << "After push: " << list.count() << endl;
    list.pop(); // List has 5,3
    cout << "After pop: " << list.count() << endl;
    list.shift(); // Remove first node (5)
    cout << "After shift: " << list.count() << endl;
    return 0;
}
#include <vector>
#include <stdexcept>

namespace circular_buffer {

using namespace circular_buffer;

template <typename ValueType>
class circular_buffer {
public:
    circular_buffer(std::size_t capacity) 
        : buffer_(capacity), head_(0), tail_(0) {}
    ValueType read() {
        if (is_empty()) throw std::domain_error("Circular buffer is empty.");
        ValueType item = buffer_[head_];
        move_position(head_);
        return item;
    }

    template <typename ValueType>
    void write(ValueType item) {
        if (is_full()) throw std::domain_error("Circular buffer is full.");
        push_back(item);
    }

    template <typename ValueType>
    void overwrite(ValueType item) {
        if (is_full()) {
            move_position(head_);
        }
        push_back(item);
    }

    template <typename ValueType>
    void clear() {
        head_ = tail_ = 0;
    }

    template <typename ValueType>
    void push_back(ValueType item) {
        buffer_[tail_] = item;
        move_position(tail_);
    }

    template <typename ValueType>
    void move_position(std::size_t& position) {
        position = (position + 1) % buffer_.size();
    }

    bool is_empty() const {
        return head_ == tail_;
    }

    bool is_full() const {
        return head_ == (tail_ + 1) % buffer_.size();
    }
};

// Example usage:
int main() {
    std::vector<ValueType> buffer_{{1, 2, 3, 4, 5, 6, 7}};
    circular_buffer<ValueType>(7) circular_buffer(buffer_);
    // ... operations on the buffer ...
}
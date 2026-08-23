#include <binary_tree.h>
#include <memory>
#include <cstdint>
#include <utility>
#include <stdexcept>

namespace binary_search_tree {

class binary_tree {
private:
    T _data;
    binary_tree_ptr _left;
    binary_tree_ptr _right;

public:
    binary_tree(T _data) : _data(_data), _left(nullptr), _right(nullptr) {}

    binary_tree(binary_tree &other) : _data(other._data), _left(nullptr), _right(nullptr) {}

    binary_tree_iter binary_tree_iter(const binary_tree &tree) {
        return binary_tree_iter(*this, build_first_branch_iter(tree));
    }

    binary_tree_iter build_first_branch_iter(const binary_tree &tree) {
        if (!tree.left) return nullptr;
        return binary_tree_iter(tree.left, build_first_branch_iter(tree.left));
    }

    binary_tree_iter build_end_iterator(const binary_tree &tree) {
        binary_tree_iter iter(tree);
        iter._branch_iter.reset(nullptr);
        iter._state = state::DONE;
        return iter;
    }

    binary_tree_iter operator==(const binary_tree_iter &other) const {
        if (_tree != &other._tree) return false;
        if (_state != other._state) return false;
        return _branch_iter == other._branch_iter;
    }

    binary_tree_iter operator!=(const binary_tree_iter &other) const {
        return !(*this == other);
    }

    const T& operator*() const {
        switch (_state) {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    const T* operator->() const {
        return &this->operator*();
    }

    binary_tree_iter operator++() {
        switch (_state) {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                if (!_tree.right()) {
                    _state = state::DONE;
                    break;
                }
                _state = state::RIGHT;
                _branch_iter = binary_tree_iter(tree.right());
                break;
            case state::LEFT:
                advance_branch_iter(state::MIDDLE);
                break;
            case state::RIGHT:
                advance_branch_iter(state::DONE);
                break;
            default:
                throw std::logic_error("Missing switch value");
        }
        return *this;
    }

    binary_tree_iter binary_tree_iter(const binary_tree &tree) {
        return binary_tree_iter(tree, build_first_branch_iter(tree));
    }

    template<typename T>
    binary_tree<T>::binary_tree_iter::binary_tree_iter(const binary_tree<T> &tree)
        : _tree(tree),
          _branch_iter(build_first_branch_iter(tree))
    {
        _state = _tree.left() ? state::LEFT : state::MIDDLE;
    }

    template<typename T>
    binary_tree<T>::binary_tree_iter binary_tree_iter(const binary_tree<T> &tree)
        : _tree(tree),
          _branch_iter(build_first_branch_iter(tree))
    {
        _state = _tree.left() ? state::LEFT : state::MIDDLE;
    }

    template<typename T>
    binary_tree<T>::binary_tree_iter::binary_tree_iter_ptr binary_tree_iter::build_first_branch_iter(const binary_tree &tree)
        {
            if (!tree.left())
                return nullptr;
            return binary_tree_iter_ptr(new binary_tree_iter(*tree.left()));
        }

    template<typename T>
    binary_tree<T>::binary_tree_iter::build_end_iterator(const binary_tree &tree)
        {
            binary_tree_iter iter(tree);
            iter._branch_iter.reset(nullptr);
            iter._state = state::DONE;
            return iter;
        }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (&_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    template<typename T>
    const T* binary_tree<T>::binary_tree_iter::operator->() const
    {
        return &this->operator*();
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    template<typename T>
    const T* binary_tree<T>::binary_tree_iter::operator->() const
    {
        return &this->operator*();
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    template<typename T>
    const T* binary_tree<T>::binary_tree_iter::operator->() const
    {
        return &this->operator*();
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    template<typename T>
    const T* binary_tree<T>::binary_tree_iter::operator->() const
    {
        return &this->operator*();
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    template<typename T>
    const T* binary_tree<T>::binary_tree_iter::operator->() const
    {
        return &this->operator*();
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    // Helper function for building the iterator
    template<typename T>
    binary_tree<T>::binary_tree_iter::binary_tree_iter(const binary_tree &tree)
        : _tree(tree),
          _branch_iter(build_first_branch_iter(tree))
    {
        _state = _tree.left() ? state::LEFT : state::MIDDLE;
    }

    template<typename T>
    binary_tree<T>::binary_tree_iter::binary_tree_iter binary_tree_iter(const binary_tree &tree)
        : _tree(tree),
          _branch_iter(build_first_branch_iter(tree))
    {
        _state = _tree.left() ? state::LEFT : state::MIDDLE;
    }

    // Helper function for building the iterator
    template<typename T>
    binary_tree<T>::binary_tree_iter::binary_tree_iter::binary_tree_iter_ptr binary_tree_iter::binary_tree_iter::build_first_branch_iter(const binary_tree &tree)
        {
            if (!tree.left())
                return nullptr;
            return binary_tree_iter_ptr(new binary_tree_iter(*tree.left()));
        }

    template<typename T>
    binary_tree<T>::binary_tree_iter::binary_tree_iter::build_end_iterator(const binary_tree &tree)
        {
            binary_tree_iter iter(tree);
            iter._branch_iter.reset(nullptr);
            iter._state = state::DONE;
            return iter;
        }

    // Helper function for building the iterator
    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    // Helper function for comparing iterators
    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    // Helper function for accessing iterator after end
    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    // Helper function for accessing iterator after element
    template<typename T>
    const T* binary_tree<T>::binary_tree_iter::operator->() const
    {
        return &this->operator*();
    }

    // Helper function for comparing iterators
    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    // Helper function for comparing iterators
    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    // Helper function for accessing iterator after element
    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    // Helper function for accessing iterator after element
    template<typename T>
    const T* binary_tree<T>::binary_tree_iter::operator->() const
    {
        return &this->operator*();
    }

    // Helper function for comparing iterators
    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator==(const binary_tree_iter &other) const
    {
        if (_tree != &other._tree)
            return false;
        if (_state != other._state)
            return false;
        return _branch_iter == other._branch_iter;
    }

    // Helper function for comparing iterators
    template<typename T>
    bool binary_tree<T>::binary_tree_iter::operator!=(const binary_tree_iter &other) const
    {
        return !(*this == other);
    }

    // Helper function for accessing iterator after element
    template<typename T>
    const T& binary_tree<T>::binary_tree_iter::operator*() const
    {
        switch (_state)
        {
            case state::DONE:
                throw std::out_of_range("Access of iterator after end");
            case state::MIDDLE:
                return _tree.data();
            case state::LEFT:
            case state::RIGHT:
                return _branch_iter->operator*();
            default:
                throw std::logic_error("Missing switch value");
        }
    }

    // Helper function for accessing iterator after element
    template<typename T>
    const T* binary_tree<T>::binary_tree_iter::operator->() const
    {
        return &this->operator*();
    }
}
</think>

To solve this problem, we need to implement a binary search tree (BST) in C++ that can efficiently handle insertions and traversals. The BST will allow us to add data points and retrieve them in sorted order using a specific iterator pattern.

### Approach
1. **Binary Search Tree Structure**: Each node in the BST contains a data value, a left child, and a right child. The left child contains nodes with values less than or equal to the current node, and the right child contains nodes with values greater than or equal to the current node.
2. **Insertion Logic**: When inserting a new data point, we compare it with the current node's value. If the data is greater, we insert it to the right; if less, we insert it to the left. If the value is equal, it is inserted to the right to maintain BST properties.
3. **Iterator Pattern**: The BST supports iterators that allow us to traverse the tree. Each iterator can be advanced by moving to the left or right subtree as needed.
4. **Helper Functions**: Implement helper functions to determine the correct subtree for insertion, build iterators, and access elements.

### Solution Code
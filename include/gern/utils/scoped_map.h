// From TACO
#pragma once

#include <list>
#include <ostream>
#include <set>

namespace gern {
namespace util {

template<typename Key, typename Value>
class ScopedMap {
public:
    ScopedMap() {
        scope();
    }

    ~ScopedMap() {
        unscope();
    }

    /// Add a level of scoping.
    void scope() {
        scopes.push_front(std::map<Key, Value>());
    }

    /// Remove a level of scoping.
    void unscope() {
        scopes.pop_front();
    }

    std::map<Key, Value> pop_front() {
        auto front = scopes.front();
        unscope();
        return front;
    }

    std::map<Key, Value> front() {
        auto front = scopes.front();
        return front;
    }

    void insert(const Key &key, const Value &value) {
        scopes.front()[key] = value;
    }

    void remove(const Key &key) {
        for (auto &scope : scopes) {
            const auto it = scope.find(key);
            if (it != scope.end()) {
                scope.erase(it);
                return;
            }
        }
    }

    bool contains_in_current_scope(const Key &key) const {
        std::map<Key, Value> current_scope = scopes.front();
        if (current_scope.find(key) != current_scope.end()) {
            return true;
        }
        return false;
    }

    bool contains(const Key &key) const {
        for (auto &scope : scopes) {
            if (scope.find(key) != scope.end()) {
                return true;
            }
        }
        return false;
    }

    Value at(const Key &key) const {
        for (auto it = scopes.begin(); it != scopes.end(); ++it) {
            const auto found = it->find(key);
            if (found != it->end()) {
                return found->second;
            }
        }
        throw std::out_of_range("Key not found in ScopedMap");
    }

    friend std::ostream &operator<<(std::ostream &os, ScopedMap<Key, Value> sset) {
        os << "ScopedMap:" << std::endl;

        for (auto &scope : sset.scopes) {
            os << "Scope{";
            for (auto &elem : scope) {
                os << elem.first << " : " << elem.second << ",";
            }
            os << "}" << std::endl;
        }

        return os;
    }

private:
    std::list<std::map<Key, Value>> scopes;
};

}  // namespace util
}  // namespace gern
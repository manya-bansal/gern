// From TACO
#pragma once

#include <list>
#include <ostream>
#include <set>

namespace gern {
namespace util {

template<typename Key>
class ScopedSet {
public:
    ScopedSet() {
        scope();
    }

    ~ScopedSet() {
        unscope();
    }

    /// Add a level of scoping.
    void scope() {
        scopes.push_front(std::set<Key>());
    }

    std::set<Key> pop() {
        auto scope = scopes.front();
        scopes.pop_front();
        return scope;
    }

    std::set<Key> front() {
        return scopes.front();
    }

    /// Remove a level of scoping.
    void unscope() {
        scopes.pop_front();
    }

    void insert(const Key &key) {
        scopes.front().insert(key);
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

    bool contains_at_current_scope(const Key &key) {
        auto scope_first = scopes.front();
        return scope_first.find(key) != scope_first.end();
    }

    bool contains(const Key &key) {
        for (auto &scope : scopes) {
            if (scope.find(key) != scope.end()) {
                return true;
            }
        }
        return false;
    }

    friend std::ostream &operator<<(std::ostream &os, ScopedSet<Key> sset) {
        os << "ScopedSet:" << std::endl;

        for (auto &scope : sset.scopes) {
            os << "Scope{";
            for (auto &elem : scope) {
                os << elem << ", ";
            }
            os << "}" << std::endl;
        }

        return os;
    }

private:
    std::list<std::set<Key>> scopes;
};

}  // namespace util
}  // namespace gern
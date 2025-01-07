#pragma once

#include <exception>
#include <iostream>
#include <string>

namespace gern {

namespace error {
class Error : public std::exception {
protected:
    std::string message;  // Error message
public:
    explicit Error(const std::string &msg)
        : message(msg) {
    }
    const char *what() const noexcept override {
        return message.c_str();
    }
};

// Derived class for user-related errors
class UserError : public Error {
public:
    explicit UserError(const std::string &msg)
        : Error("User Error: " + msg) {
    }
};

class InternalError : public Error {
public:
    explicit InternalError(const std::string &msg)
        : Error("Internal Error: " + msg) {
    }
};

}  // namespace error
}  // namespace gern
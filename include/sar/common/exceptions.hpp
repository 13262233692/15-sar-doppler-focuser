#pragma once

#include <stdexcept>
#include <string>
#include <source_location>

namespace sar {

class SARException : public std::runtime_error {
public:
    explicit SARException(
        const std::string& msg,
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_message(msg, loc)) {}

private:
    static std::string format_message(
        const std::string& msg,
        const std::source_location& loc) {
        return "[" + std::string(loc.file_name()) + ":" +
               std::to_string(loc.line()) + "][" +
               std::string(loc.function_name()) + "] " + msg;
    }
};

class IOException : public SARException {
public:
    using SARException::SARException;
};

class MemoryException : public SARException {
public:
    using SARException::SARException;
};

class InvalidParameterException : public SARException {
public:
    using SARException::SARException;
};

}

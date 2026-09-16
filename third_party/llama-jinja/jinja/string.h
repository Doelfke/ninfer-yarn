#pragma once

#include "utils.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace jinja {

// Origins are opaque input-region identifiers. They never affect rendering or tokenization.
struct string_part {
    std::uint32_t origin = 0;
    std::string val;
    std::optional<std::size_t> source_offset = 0;
};

struct string {
    std::vector<string_part> parts;
    string() = default;

    string(const std::string& value, std::uint32_t origin = 0) : parts{{origin, value, 0}} {}

    string(int value) : string(std::to_string(value)) {}

    string(double value) : string(std::to_string(value)) {}

    void tag(std::uint32_t origin, bool exact = true);
    std::string str() const;
    std::size_t length() const;
    std::size_t byte_size() const;
    void hash_update(hasher& hash) const noexcept;
    bool is_uppercase() const;
    bool is_lowercase() const;
    string& append(const string& other);
    string cut_bytes(std::size_t begin, std::size_t end) const;
    string slice(std::optional<std::int64_t> start, std::optional<std::int64_t> stop,
                 std::int64_t step = 1) const;
    std::vector<string> split(const std::optional<std::string>& separator, int64_t maxsplit,
                              bool reverse) const;
    string transformed(std::string text) const;
    string uppercase() const;
    string lowercase() const;
    string capitalize() const;
    string titlecase() const;
    string strip(bool left, bool right,
                 std::optional<const std::string_view> chars = std::nullopt) const;
};

} // namespace jinja

#include "text/jinja.h"

#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>

namespace {
using Json = nlohmann::ordered_json;
using ninfer::text::JinjaTemplate;

int check(bool condition, const char* message) {
    if (condition) return 0;
    std::cerr << message << '\n';
    return 1;
}

int language_semantics() {
    struct Example {
        std::string source;
        Json input;
        std::string expected;
    };

    const Example cases[]{
        {"{{ missing is undefined }}|{{ value is none }}|{{ value|default('fallback') }}|{{ "
         "value|default('fallback', true) }}",
         {{"value", nullptr}},
         "True|True|None|fallback"},
        {"{{ missing }}|{{ [1, none, true, 'x'] }}|{{ 12 ~ none }}", Json::object(),
         "|[1, None, True, 'x']|12None"},
        {"{{ text|length }}|{{ text[-1] }}|{{ text[::-1] }}|{{ text[:-1:-1] }}|{{ text[99:] }}",
         {{"text", "北京🌏"}},
         "3|🌏|🌏京北||"},
        {"{% for c in text %}[{{ c }}]{% endfor %}", {{"text", "北京🌏"}}, "[北][京][🌏]"},
        {"{{ text|upper }}|{{ text|lower }}|{{ text|capitalize }}",
         {{"text", "straße ΟΣ İ"}},
         "STRASSE ΟΣ İ|straße ος i̇|Straße ος i̇"},
        {"{{ text|trim }}|{{ text.split()|tojson }}|{{ text.rsplit(none,1)|tojson }}",
         {{"text", "\u0085北京\u3000two\tend\u001c"}},
         "北京　two\tend|[\"北京\", \"two\", \"end\"]|[\"\u0085北京　two\", \"end\"]"},
        {"{{ text.replace('', '-', 2) }}|{{ text.replace('京', 'x', 1) }}",
         {{"text", "北京🌏"}},
         "-北-京🌏|北x🌏"},
        {"{{ value|tojson(ensure_ascii=true,sort_keys=true,separators=(',',':')) }}",
         {{"value", {{"z", "🌏"}, {"a", false}}}},
         "{\"a\":false,\"z\":\"\\ud83c\\udf0f\"}"},
        {"{% macro show(x) %}{{ x|default('local') }}{% endmacro %}{% set ns=namespace(n=0) %}{% "
         "for x in [1,2] %}{% set ns.n=ns.n+x %}{% endfor %}{{ ns.n }}|{{ show() }}",
         Json::object(), "3|local"},
        {"{% set x='outer' %}{% macro show(a='A',b='B') %}{{ x }}:{{ a }}:{{ b }}{% endmacro %}{% "
         "for x in ['inner'] %}{{ show(b='Z') }}{% endfor %}",
         Json::object(), "outer:A:Z"},
        {"{% set n=namespace(f=none) %}{% for x in ['captured'] %}{% macro show() %}{{ x }}{% "
         "endmacro %}{% set n.f=show %}{% endfor %}{{ n.f() }}",
         Json::object(), "captured"},
        {"{% macro wrap() %}[{{ caller('value') }}]{% endmacro %}{% call(x) wrap() %}{{ x }}{% "
         "endcall %}",
         Json::object(), "[value]"},
        {"{{ number }}", {{"number", 1.23456789}}, "1.23456789"},
        {"{% filter upper %}{{ text }}{% endfilter %}", {{"text", "straße"}}, "STRASSE"},
    };
    int failures = 0;
    for (const auto& item : cases) {
        const auto actual = JinjaTemplate(item.source, "language-test").render(item.input).text;
        if (actual != item.expected) {
            std::cerr << "Jinja semantic mismatch in " << item.source
                      << "\nexpected: " << Json(item.expected).dump()
                      << "\nactual: " << Json(actual).dump() << '\n';
            ++failures;
        }
    }
    for (const char* source : {"{{ none + 'x' }}", "{{ 'abc'[::0] }}", "{{ missing|tojson }}",
                               "{{ value|unknown_filter }}"}) {
        try {
            (void)JinjaTemplate(source, "error-test").render(Json::object());
            ++failures;
        } catch (const std::invalid_argument& error) {
            failures += check(std::string(error.what()).starts_with("error-test:"),
                              "template error omitted its source name");
        }
    }
    return failures;
}

int origins_and_requests() {
    const JinjaTemplate compiled(
        "{% filter upper %}{{ text|trim }}{% endfilter %}|{{ tool|tojson }}", "shared-test");
    const Json context{{"text", " hello "}, {"tool", {{"name", "f"}}}};
    const std::vector<ninfer::text::TemplateInputRegion> regions{{"/text", 7}, {"/tool", 9}};
    ninfer::text::TemplateRenderOptions options{.timestamp = 0, .regions = regions};
    const auto result = compiled.render(context, options);
    int failures = check(result.text == "HELLO|{\"name\": \"f\"}" && result.regions.size() == 2 &&
                             result.regions[0].tag == 7 && result.regions[0].begin == 0 &&
                             result.regions[0].end == 5 && !result.regions[0].source_offset &&
                             result.regions[1].tag == 9 && result.regions[1].begin == 6 &&
                             result.regions[1].end == result.text.size(),
                         "transformed/JSON input regions were lost or incorrectly mapped");
    failures += check(compiled.render(context).regions.empty(),
                      "unannotated render retained request origins");
    const auto clipped =
        JinjaTemplate("{{ text[1:3] }}", "slice-test").render({{"text", "a北京c"}}, options);
    failures += check(clipped.regions.size() == 1 && clipped.regions[0].source_offset == 1 &&
                          clipped.regions[0].end == 6,
                      "Unicode slice lost byte origin mapping");
    std::vector<std::future<bool>> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.push_back(std::async(std::launch::async, [&] {
            for (int i = 0; i < 100; ++i)
                if (compiled.render(context, options).text != result.text) return false;
            return true;
        }));
    }
    for (auto& worker : workers)
        failures += check(worker.get(), "shared template leaked mutable request state");

    struct Cancelled : std::runtime_error {
        Cancelled() : std::runtime_error("cancelled") {}
    };

    int calls          = 0;
    options.checkpoint = [&] {
        if (++calls > 2) throw Cancelled();
    };
    try {
        (void)JinjaTemplate("{% for i in range(1000000) %}{{ i }}{% endfor %}", "cancel-test")
            .render(Json::object(), options);
        failures += check(false, "template ignored cancellation");
    } catch (const Cancelled&) {}
    options.checkpoint = {};
    const JinjaTemplate clock("{{ strftime_now('%Y') }}", "clock-test");
    failures += check(clock.render(Json::object(), options).text ==
                          clock.render(Json::object(), options).text,
                      "time snapshot was not reused");
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--render") {
        for (std::string line; std::getline(std::cin, line);) {
            Json result;
            try {
                const auto input = Json::parse(line);
                result["text"]   = JinjaTemplate(input.at("source"), "reference-case")
                                     .render(input.at("context"))
                                     .text;
                result["ok"] = true;
            } catch (const std::exception& error) {
                result["ok"]    = false;
                result["error"] = error.what();
            }
            std::cout << result.dump() << '\n';
        }
        return 0;
    }
    try {
        return language_semantics() + origins_and_requests() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

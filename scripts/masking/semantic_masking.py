#!/usr/bin/env python3

import re


NUMERIC_SPAN_RE = re.compile(
    r"""
    (?P<latex_frac>
        \\[dDtT]?frac
        \s*\{[^{}]*?\d[^{}]*?\}
        \s*\{[^{}]*?\d[^{}]*?\}
    )
    |
    (?P<sqrt>
        \\sqrt
        (?:\s*\[[^\[\]]*?\d[^\[\]]*?\])?
        \s*\{[^{}]*?\d[^{}]*?\}
    )
    |
    (?P<sci_latex>
        [+-]?(?:\d+\.\d+|\d+|\.\d+)
        \s*\\times\s*
        10
        \s*\^\s*\{?[+-]?\d+\}?
    )
    |
    (?P<fraction>
        [+-]?(?:\d+\.\d+|\d+|\.\d+)
        \s*/\s*
        [+-]?(?:\d+\.\d+|\d+|\.\d+)
    )
    |
    (?P<percent>
        [+-]?(?:\d+\.\d+|\d+|\.\d+)
        \s*%
    )
    |
    (?P<sci>
        [+-]?(?:\d+\.\d+|\d+|\.\d+)
        [eE][+-]?\d+
    )
    |
    (?P<decimal>
        [+-]?(?:\d+\.\d+|\.\d+)
    )
    |
    (?P<integer>
        [+-]?\d+
    )
    """,
    re.VERBOSE,
)


STRUCTURAL_NUMBER_RE = re.compile(
    r"""
    \b
    (?P<label>
        step
        |case
        |part
        |subcase
        |problem
        |example
        |exercise
        |lemma
        |theorem
        |claim
        |stage
        |round
    )
    \s+
    (?P<num>\d+)
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def mask_math_teacher_trace(text: str) -> str:
    protected = {}

    def protect_structural(match):
        key = f"__STRUCTURAL_NUMBER_{len(protected)}__"
        protected[key] = match.group(0)
        return key

    masked_text = STRUCTURAL_NUMBER_RE.sub(protect_structural, text)
    masked_text = NUMERIC_SPAN_RE.sub("[NUMBER]", masked_text)
    for key, value in protected.items():
        masked_text = masked_text.replace(key, value)
    return masked_text


def mask_code_teacher_trace(text: str) -> str:
    pattern = r"```([^\n`]*)\n.*?```"

    def repl(match):
        lang = match.group(1).strip()
        if lang:
            return f"```{lang}\n[CODE]\n```"
        return "```\n[CODE]\n```"

    return re.sub(pattern, repl, text, flags=re.DOTALL)

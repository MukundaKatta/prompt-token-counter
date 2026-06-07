"""Tests for prompt_token_counter."""

from __future__ import annotations

import pytest

from prompt_token_counter import BudgetResult, TokenCounter, chars_per_4_estimate


# ---- chars_per_4_estimate -------------------------------------------


def test_chars_per_4_empty_is_zero():
    assert chars_per_4_estimate("") == 0


def test_chars_per_4_basic():
    # "abcd" = 1 token; "abcde" = 2 (ceil)
    assert chars_per_4_estimate("abcd") == 1
    assert chars_per_4_estimate("abcde") == 2
    assert chars_per_4_estimate("a" * 100) == 25


# ---- count single string -------------------------------------------


def test_count_empty_string_zero():
    c = TokenCounter()
    assert c.count("") == 0


def test_count_uses_chars_per_4_by_default():
    c = TokenCounter()
    assert c.count("a" * 100) == 25


def test_count_uses_tokenize_callable_returning_list():
    c = TokenCounter(tokenize=lambda s: list(s))  # 1 token per char
    assert c.count("hello") == 5


def test_count_uses_tokenize_callable_returning_int():
    c = TokenCounter(tokenize=lambda s: 42)
    assert c.count("hello") == 42


def test_count_normalizes_bool_tokenize_return_to_int():
    # bool is a subclass of int; count() must return a plain int, not a bool.
    c = TokenCounter(tokenize=lambda s: True)
    out = c.count("hello")
    assert out == 1
    assert type(out) is int


def test_count_invalid_tokenize_return_raises():
    c = TokenCounter(tokenize=lambda s: 3.14)  # not int, not sized
    with pytest.raises(TypeError):
        c.count("hi")


# ---- count_messages -----------------------------------------------


def test_count_messages_plain_text():
    c = TokenCounter(per_message_overhead=0)
    out = c.count_messages(
        [
            {"role": "user", "content": "a" * 100},
        ]
    )
    assert out == 25  # 100/4


def test_count_messages_includes_per_message_overhead():
    c = TokenCounter(per_message_overhead=10)
    out = c.count_messages(
        [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]
    )
    assert out == 20


def test_count_messages_text_block_content():
    c = TokenCounter(per_message_overhead=0)
    out = c.count_messages(
        [
            {"role": "user", "content": [{"type": "text", "text": "a" * 40}]},
        ]
    )
    assert out == 10  # 40/4


def test_count_messages_image_block_flat_overhead():
    c = TokenCounter(per_message_overhead=0)
    out = c.count_messages(
        [
            {"role": "user", "content": [{"type": "image", "source": {}}]},
        ]
    )
    assert out == 256


def test_count_messages_tool_use_block():
    c = TokenCounter(per_message_overhead=0)
    out = c.count_messages(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "u",
                        "name": "search",
                        "input": {"q": "x"},
                    },
                ],
            },
        ]
    )
    # name="search" → 2 tokens, input='{"q":"x"}' → 3 tokens, +8 overhead = 13
    assert out > 0


def test_count_messages_tool_result_block_string_content():
    c = TokenCounter(per_message_overhead=0)
    out = c.count_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "u", "content": "a" * 40},
                ],
            },
        ]
    )
    assert out == 14  # 10 + 4 overhead


def test_count_messages_skips_none_content():
    c = TokenCounter(per_message_overhead=0)
    out = c.count_messages([{"role": "assistant", "content": None}])
    assert out == 0


def test_count_messages_unknown_content_uses_repr():
    c = TokenCounter(per_message_overhead=0)
    out = c.count_messages([{"role": "user", "content": 42}])
    assert out > 0


# ---- count_tools --------------------------------------------------


def test_count_tools_basic():
    c = TokenCounter(per_tool_overhead=0)
    out = c.count_tools(
        [
            {
                "name": "search",
                "description": "search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ]
    )
    assert out > 0


def test_count_tools_per_tool_overhead():
    c = TokenCounter(per_tool_overhead=100)
    out = c.count_tools([{"name": "", "description": "", "input_schema": {}}])
    assert out >= 100


def test_count_tools_handles_openai_parameters_key():
    c = TokenCounter(per_tool_overhead=0)
    out = c.count_tools(
        [
            {"name": "f", "description": "d", "parameters": {"type": "object"}},
        ]
    )
    assert out > 0


# ---- fits ---------------------------------------------------------


def test_fits_returns_budget_result():
    c = TokenCounter(per_message_overhead=0)
    out = c.fits(
        context_window=1000,
        system="a" * 40,  # 10 tokens
        messages=[{"role": "user", "content": "b" * 40}],  # 10 tokens
    )
    assert isinstance(out, BudgetResult)
    assert out.used == 20
    assert out.remaining == 980
    assert out.fits is True


def test_fits_handles_system_as_block_list():
    c = TokenCounter(per_message_overhead=0)
    out = c.fits(
        context_window=1000,
        system=[{"type": "text", "text": "a" * 40}],
    )
    assert out.used == 10


def test_fits_subtracts_reserved_output():
    c = TokenCounter(per_message_overhead=0)
    out = c.fits(
        context_window=100,
        system="a" * 400,  # 100 tokens
        reserved_output=50,
    )
    # effective budget = 50, used = 100 → does not fit
    assert out.fits is False
    assert out.remaining < 0


def test_fits_doesnt_under_zero_when_reserved_too_big():
    c = TokenCounter()
    out = c.fits(context_window=100, reserved_output=200)
    assert out.context_window == 100
    assert out.reserved_output == 200


def test_fits_all_optional_args_none():
    c = TokenCounter()
    out = c.fits(context_window=1000)
    assert out.used == 0
    assert out.remaining == 1000


def test_fits_includes_tool_tokens():
    c = TokenCounter(per_tool_overhead=10)
    out = c.fits(
        context_window=1000,
        tools=[{"name": "f", "description": "", "input_schema": {}}],
    )
    assert out.used >= 10


# ---- end-to-end with BYO tokenizer ---------------------------------


def test_byo_tokenizer_used_for_messages_and_tools():
    # 1 token per char
    c = TokenCounter(
        tokenize=lambda s: list(s), per_message_overhead=0, per_tool_overhead=0
    )
    out = c.count_messages([{"role": "user", "content": "hello"}])
    assert out == 5

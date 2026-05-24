"""prompt-token-counter - approximate token counts for LLM payloads.

Most agent code needs to answer: "Will this request fit in the model's
context window?" The honest answers are vendor-specific (tiktoken for
OpenAI, Anthropic's counts API for Claude, sentencepiece for Llama).
The cheap-and-acceptable answer is `chars/4`, which is within ~15% for
English prose.

`TokenCounter` wraps either approach behind one API:

    from prompt_token_counter import TokenCounter

    # zero deps, chars/4 heuristic
    counter = TokenCounter()

    counter.count("Hello world")
    counter.count_messages([
        {"role": "system", "content": "..."},
        {"role": "user", "content": [{"type": "text", "text": "..."}]},
    ])
    counter.count_tools([{"name": "search", "description": "...", "input_schema": {...}}])

    fits = counter.fits(
        context_window=200_000,
        system="...",
        messages=[...],
        tools=[...],
        reserved_output=4096,
    )

    # exact: BYO tokenizer
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    counter = TokenCounter(tokenize=lambda s: enc.encode(s))

`tokenize` is `Callable[[str], list[int] | int]` — return either a list
of token ids (the lib will count them) or an integer count directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

__version__ = "0.1.0"
__all__ = [
    "TokenCounter",
    "BudgetResult",
    "chars_per_4_estimate",
]


TokenizeFn = Callable[[str], Any]


def chars_per_4_estimate(text: str) -> int:
    """Default token estimate: ceil(len / 4). Within ~15% for English prose."""
    if not text:
        return 0
    n = len(text)
    return -(-n // 4)  # ceil division


@dataclass(frozen=True)
class BudgetResult:
    """Outcome of `TokenCounter.fits(...)`."""

    fits: bool
    used: int
    remaining: int
    context_window: int
    reserved_output: int


class TokenCounter:
    """Approximate token counts for LLM messages, system prompts, and tools.

    Args:
        tokenize: optional `(str) -> int | list[int]`. If a callable is
            supplied, it owns counting. Otherwise the chars/4 estimate runs.
            Useful values:
              - tiktoken: `lambda s: enc.encode(s)` returns list[int]
              - Anthropic's token-counting endpoint: write a wrapper
                that returns the int count from the API
        per_message_overhead: tokens to add per message (covers
            role/delimiter framing). Default 4 (rough Anthropic average).
        per_tool_overhead: tokens to add per tool (covers JSON wrapping).
            Default 8.
    """

    def __init__(
        self,
        tokenize: TokenizeFn | None = None,
        *,
        per_message_overhead: int = 4,
        per_tool_overhead: int = 8,
    ) -> None:
        self._tokenize = tokenize
        self._per_message = int(per_message_overhead)
        self._per_tool = int(per_tool_overhead)

    # ---- single string ----------------------------------------------

    def count(self, text: str) -> int:
        """Tokens in a plain string."""
        if not text:
            return 0
        if self._tokenize is None:
            return chars_per_4_estimate(text)
        result = self._tokenize(text)
        if isinstance(result, int):
            return result
        try:
            return len(result)
        except TypeError as exc:
            raise TypeError(
                "tokenize callable must return int or sized iterable; "
                f"got {type(result).__name__}"
            ) from exc

    # ---- message list ------------------------------------------------

    def count_messages(self, messages: Iterable[dict[str, Any]]) -> int:
        """Tokens in a list of role+content messages.

        Handles plain `{"role": ..., "content": "..."}` and the Anthropic
        content-block shape `{"role": ..., "content": [{"type": "text",
        "text": "..."}, ...]}` (and tool_use, tool_result, image blocks
        with sensible fallbacks).
        """
        total = 0
        for msg in messages:
            content = msg.get("content")
            total += self._per_message
            if isinstance(content, str):
                total += self.count(content)
                continue
            if isinstance(content, list):
                for block in content:
                    total += self._count_block(block)
                continue
            if content is None:
                continue
            # unknown shape — fall back to repr
            total += self.count(repr(content))
        return total

    # ---- tools ------------------------------------------------------

    def count_tools(self, tools: Iterable[dict[str, Any]]) -> int:
        """Tokens in a list of Anthropic-shape tool definitions."""
        import json

        total = 0
        for tool in tools:
            total += self._per_tool
            # serialize the relevant fields and count them
            name = tool.get("name", "")
            desc = tool.get("description", "")
            schema = tool.get("input_schema") or tool.get("parameters") or {}
            total += self.count(name)
            total += self.count(desc)
            try:
                schema_str = json.dumps(schema, separators=(",", ":"))
            except (TypeError, ValueError):
                schema_str = repr(schema)
            total += self.count(schema_str)
        return total

    # ---- fits / budget ----------------------------------------------

    def fits(
        self,
        *,
        context_window: int,
        system: str | list[dict[str, Any]] | None = None,
        messages: Iterable[dict[str, Any]] | None = None,
        tools: Iterable[dict[str, Any]] | None = None,
        reserved_output: int = 0,
    ) -> BudgetResult:
        """Tally everything and return a `BudgetResult`.

        `system` may be a string OR a list of content blocks (the Anthropic
        prompt-block shape). Other args are optional. `reserved_output` is
        subtracted from `context_window` before the fit check.
        """
        used = 0
        if isinstance(system, str):
            used += self.count(system)
        elif isinstance(system, list):
            for block in system:
                used += self._count_block(block)
        if messages is not None:
            used += self.count_messages(messages)
        if tools is not None:
            used += self.count_tools(tools)
        effective_budget = max(context_window - reserved_output, 0)
        return BudgetResult(
            fits=used <= effective_budget,
            used=used,
            remaining=effective_budget - used,
            context_window=context_window,
            reserved_output=reserved_output,
        )

    # ---- internals --------------------------------------------------

    def _count_block(self, block: Any) -> int:
        """Count a single content block (text/image/tool_use/tool_result/...)."""
        if isinstance(block, str):
            return self.count(block)
        if not isinstance(block, dict):
            return self.count(repr(block))
        btype = block.get("type")
        if btype == "text":
            return self.count(block.get("text", ""))
        if btype == "image":
            # Treat each image as a flat overhead — vendor-specific in reality.
            return 256
        if btype == "tool_use":
            import json

            name = block.get("name", "")
            try:
                input_str = json.dumps(block.get("input", {}), separators=(",", ":"))
            except (TypeError, ValueError):
                input_str = repr(block.get("input"))
            return self.count(name) + self.count(input_str) + 8
        if btype == "tool_result":
            content = block.get("content")
            if isinstance(content, str):
                return self.count(content) + 4
            if isinstance(content, list):
                return sum(self._count_block(b) for b in content) + 4
            return self.count(repr(content)) + 4
        if btype == "document":
            return 1024  # placeholder flat overhead
        # unknown type — repr fallback
        return self.count(repr(block))

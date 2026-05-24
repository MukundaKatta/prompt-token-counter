# prompt-token-counter

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/prompt-token-counter.svg)](https://pypi.org/project/prompt-token-counter/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Approximate token counts for LLM messages, system prompts, and tools.** Zero deps, BYO tokenizer.

```python
from prompt_token_counter import TokenCounter

# zero deps: chars/4 heuristic, ~15% accurate for English prose
counter = TokenCounter()

counter.count("Hello world")
counter.count_messages([
    {"role": "system", "content": "..."},
    {"role": "user", "content": [{"type": "text", "text": "..."}]},
])
counter.count_tools([{
    "name": "search",
    "description": "search the web",
    "input_schema": {"type": "object", "properties": {...}},
}])

result = counter.fits(
    context_window=200_000,
    system=long_system_prompt,
    messages=conversation,
    tools=tool_defs,
    reserved_output=4096,
)
# BudgetResult(fits=True, used=12345, remaining=183559, context_window=200000, reserved_output=4096)
```

**Need exact counts?** Pass a `tokenize` callable.

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
counter = TokenCounter(tokenize=lambda s: enc.encode(s))
```

The callable can return a `list[int]` (the lib counts the length) or an integer directly (e.g. wrap Anthropic's `count_tokens` endpoint).

## Why

Three quarters of agent code lives downstream of one question: "Will this fit?" The honest answers are vendor-specific. The cheap answer (`chars/4`) is within ~15% for English prose and good enough for budget gates that fail open at 10–20% headroom.

`prompt-token-counter` handles the bookkeeping — sums system + messages + tools, accounts for content blocks (text, image, tool_use, tool_result, document), subtracts reserved output — behind one API that swaps cheap-vs-exact via a callable.

## Install

```bash
pip install prompt-token-counter
```

## API

```python
TokenCounter(
    tokenize=None,                  # Callable[[str], int | list[int]] | None
    per_message_overhead=4,         # tokens added per message
    per_tool_overhead=8,            # tokens added per tool
)

# atoms
counter.count(text)
counter.count_messages([{role, content}, ...])
counter.count_tools([{name, description, input_schema}, ...])

# all-in-one
counter.fits(
    *,
    context_window: int,
    system: str | list[dict] | None = None,
    messages: Iterable[dict] | None = None,
    tools: Iterable[dict] | None = None,
    reserved_output: int = 0,
) -> BudgetResult

BudgetResult(fits, used, remaining, context_window, reserved_output)
```

## Companion libraries

- [`prompt-cache-warmer`](https://github.com/MukundaKatta/prompt-cache-warmer) — warm system prompts whose token cost you just measured.
- [`llm-stop-conditions`](https://github.com/MukundaKatta/llm-stop-conditions) — `MaxTokens` condition consumes counts from here.
- [`agentfit`](https://github.com/MukundaKatta/agentfit) — once you know you don't fit, truncate to fit.

## License

MIT

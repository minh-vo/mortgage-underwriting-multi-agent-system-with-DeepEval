"""Claude helpers: one client per run, a structured-output call and a tool loop."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel, ValidationError

from .config import get_settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000

_client: ContextVar[anthropic.AsyncAnthropic | None] = ContextVar("client", default=None)


class LLMError(RuntimeError):
    """A Claude call that did not produce a usable answer."""


@asynccontextmanager
async def use_client():
    """Open a Claude client for the duration of one run and close it afterwards.

    A client is never kept at module level, so it is never shared between event loops.
    """
    settings = get_settings()
    async with anthropic.AsyncAnthropic(api_key=settings.api_key or None,
                                        timeout=180.0, max_retries=1) as client:
        token = _client.set(client)
        try:
            yield client
        finally:
            _client.reset(token)


async def _request(agent: str, system: str, messages: list, effort: str,
                   tools: list | None = None, schema: type[BaseModel] | None = None,
                   model: str | None = None):
    """One Messages API call. Returns (response, usage record)."""
    client = _client.get()
    if client is None:
        raise LLMError("No Claude client is open. Call this inside llm.use_client().")
    output_config: dict = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema",
                                   "schema": anthropic.transform_schema(schema)}
    kwargs: dict = {}
    if tools:
        kwargs.update(tools=tools, tool_choice={"type": "auto"})
    response = await client.beta.messages.create(
        model=model or get_settings().model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
        thinking={"type": "adaptive"},
        output_config=output_config,
        betas=[FALLBACK_BETA],
        fallbacks="default",  # a refused request is retried on another model, server-side
        **kwargs,
    )
    usage = {"agent": agent, "model": response.model,
             "input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    return response, usage


def _text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def _stop_problem(response) -> str | None:
    if response.stop_reason == "refusal":
        return "The model declined the request (stop_reason: refusal)."
    if response.stop_reason == "max_tokens":
        return "The answer was cut off (stop_reason: max_tokens)."
    return None


async def structured(system: str, user: str, schema: type[BaseModel], effort: str = "high",
                     agent: str = "", model: str | None = None) -> tuple[BaseModel, list[dict]]:
    """Ask for an answer that matches `schema`. Returns (parsed answer, usage records)."""
    response, usage = await _request(agent, system, [{"role": "user", "content": user}],
                                     effort, schema=schema, model=model)
    problem = _stop_problem(response)
    if problem:
        raise LLMError(f"{agent or 'call'}: {problem}")
    try:
        return schema.model_validate_json(_text(response)), [usage]
    except ValidationError as error:
        raise LLMError(f"{agent or 'call'}: the answer did not match the schema: {error}") from error


async def text(system: str, user: str, effort: str = "high", agent: str = "",
               model: str | None = None) -> tuple[str, list[dict]]:
    """Ask for a plain-text answer. Returns (text, usage records)."""
    response, usage = await _request(agent, system, [{"role": "user", "content": user}],
                                     effort, model=model)
    problem = _stop_problem(response)
    if problem:
        raise LLMError(f"{agent or 'call'}: {problem}")
    return _text(response), [usage]


@dataclass
class ToolLoopResult:
    status: str  # "complete" or "failed"
    output: BaseModel | None = None  # parsed final answer when a schema was given
    text: str = ""
    error: str = ""
    trace: list[dict] = field(default_factory=list)  # {tool, input, output, is_error}
    usage: list[dict] = field(default_factory=list)


async def tool_loop(system: str, user: str, tools: list[dict], call_tool,
                    schema: type[BaseModel] | None = None, max_rounds: int = 8,
                    effort: str = "medium", agent: str = "") -> ToolLoopResult:
    """Let the model call tools until it gives a final answer.

    `call_tool(name, args)` is an async function returning text; an exception becomes an
    error result the model can see. A failure of any kind returns status "failed", never
    a placeholder answer.
    """
    result = ToolLoopResult(status="failed")
    messages: list = [{"role": "user", "content": user}]

    async def run(block) -> dict:
        try:
            output, is_error = await call_tool(block.name, block.input), False
        except Exception as error:  # the model sees the error and can correct itself
            output, is_error = f"Error: {error}", True
        result.trace.append({"tool": block.name, "input": block.input,
                             "output": output, "is_error": is_error})
        return {"type": "tool_result", "tool_use_id": block.id,
                "content": output, "is_error": is_error}

    for _ in range(max_rounds):
        try:
            response, usage = await _request(agent, system, messages, effort,
                                             tools=tools, schema=schema)
        except (anthropic.APIError, LLMError) as error:
            result.error = f"Claude call failed: {error}"
            return result
        result.usage.append(usage)
        problem = _stop_problem(response)
        if problem:
            result.error = problem
            return result

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        # The whole content goes back, thinking blocks included.
        messages.append({"role": "assistant", "content": response.content})
        if tool_uses:
            results = await asyncio.gather(*(run(block) for block in tool_uses))
            messages.append({"role": "user", "content": list(results)})
            continue
        if response.stop_reason == "pause_turn":
            continue

        result.text = _text(response)
        if schema is not None:
            try:
                result.output = schema.model_validate_json(result.text)
            except ValidationError as error:
                result.error = f"The final answer did not match the schema: {error}"
                return result
        result.status = "complete"
        return result

    result.error = f"No final answer after {max_rounds} rounds of tool calls."
    return result

"""llm.py against a scripted client: request shape, tool loop behaviour, failures."""

from types import SimpleNamespace as NS

import pytest

from underwriter import llm
from underwriter.models import AnalysisDraft, SupervisorPick

DRAFT = AnalysisDraft(summary="s", findings=["f"], policy_refs=["2.5"], risk_level="LOW",
                      conditions=[], recommendation="r").model_dump_json()


def text(value):
    return NS(type="text", text=value)


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


def response(*content, stop="end_turn"):
    return NS(content=list(content), stop_reason=stop, model="claude-opus-5-5",
              usage=NS(input_tokens=10, output_tokens=5))


class ScriptedClient:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def client_for():
    def make(*responses):
        client = ScriptedClient(*responses)
        llm._client.set(client)
        return client

    yield make
    llm._client.set(None)


async def test_structured_parses_the_answer_and_sends_the_required_request_shape(client_for):
    client = client_for(response(text('{"action": "critic", "areas": [], "reason": "ready"}')))
    pick, usage = await llm.structured("sys", "user", SupervisorPick, agent="supervisor")
    assert pick.action == "critic"
    assert usage == [{"agent": "supervisor", "model": "claude-opus-5-5",
                      "input_tokens": 10, "output_tokens": 5}]
    request = client.requests[0]
    assert request["betas"] == ["server-side-fallback-2026-07-01"] and request["fallbacks"] == "default"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "high"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in request


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
async def test_structured_raises_a_clear_error_on_refusal_or_cutoff(client_for, stop):
    client_for(response(stop=stop))
    with pytest.raises(llm.LLMError, match=stop):
        await llm.structured("sys", "user", SupervisorPick)


async def test_calls_fail_clearly_outside_use_client():
    with pytest.raises(llm.LLMError, match="use_client"):
        await llm.structured("sys", "user", SupervisorPick)


async def test_tool_loop_runs_tools_and_returns_all_results_in_one_message(client_for):
    thinking = NS(type="thinking", thinking="", signature="sig")
    first = response(thinking, tool_use("a", "dti_ratio", {"x": 1}),
                     tool_use("b", "broken", {}), stop="tool_use")
    client = client_for(first, response(text(DRAFT)))

    async def call_tool(name, args):
        if name == "broken":
            raise RuntimeError("boom")
        return "46.11% back-end DTI"

    result = await llm.tool_loop("sys", "user", [{"name": "dti_ratio"}], call_tool,
                                 schema=AnalysisDraft, agent="income")
    assert result.status == "complete" and result.output.risk_level == "LOW"
    assert [(c["tool"], c["is_error"]) for c in result.trace] == [("dti_ratio", False), ("broken", True)]
    assert len(result.usage) == 2

    second = client.requests[1]["messages"]
    assert second[1] == {"role": "assistant", "content": first.content}  # thinking block kept
    results = second[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] is True and "boom" in results[1]["content"]
    assert client.requests[0]["tool_choice"] == {"type": "auto"}
    assert client.requests[0]["output_config"]["effort"] == "medium"


async def test_tool_loop_fails_when_the_round_cap_is_hit(client_for):
    """P5: no placeholder answer is ever returned as a finished analysis."""
    client_for(*[response(tool_use(str(i), "dti_ratio", {}), stop="tool_use") for i in range(3)])

    async def call_tool(name, args):
        return "ok"

    result = await llm.tool_loop("sys", "user", [], call_tool, schema=AnalysisDraft, max_rounds=3)
    assert result.status == "failed" and result.output is None
    assert "after 3 rounds" in result.error


async def test_tool_loop_fails_on_an_answer_that_does_not_match_the_schema(client_for):
    client_for(response(text('{"summary": "only this"}')))
    result = await llm.tool_loop("sys", "user", [], None, schema=AnalysisDraft)
    assert result.status == "failed" and "did not match the schema" in result.error


async def test_tool_loop_fails_on_refusal(client_for):
    client_for(response(stop="refusal"))
    result = await llm.tool_loop("sys", "user", [], None, schema=AnalysisDraft)
    assert result.status == "failed" and "refusal" in result.error

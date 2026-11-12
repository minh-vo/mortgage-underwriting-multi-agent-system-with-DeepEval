"""The MCP server through a real stdio session (a local subprocess, no network)."""

import pytest

from underwriter import metrics
from underwriter.config import load_application
from underwriter.mcp_client import tool_session


async def test_server_lists_nine_tools_as_anthropic_definitions():
    async with tool_session() as tools:
        definitions = await tools.anthropic_tools()
    names = {d["name"] for d in definitions}
    assert names == {fn.__name__ for fn in metrics.CALCULATORS} | {"get_policy", "search_policy"}
    dti = next(d for d in definitions if d["name"] == "dti_ratio")
    assert set(dti) == {"name", "description", "input_schema"}
    assert set(dti["input_schema"]["properties"]) == {"monthly_debt", "monthly_income"}
    assert "INCLUDING" in dti["description"]


async def test_calculators_give_the_same_answer_over_mcp_as_in_process():
    inputs = metrics.tool_inputs(load_application("CASE-004"))
    async with tool_session() as tools:
        for fn in metrics.CALCULATORS:
            assert await tools.call(fn.__name__, inputs[fn.__name__]) == fn(**inputs[fn.__name__])


async def test_policy_tools_return_section_numbers():
    async with tool_session() as tools:
        assert (await tools.call("get_policy", {"section": "2.5"})).startswith("[Policy 2.5")
        assert "[Policy 1.2" in await tools.call("search_policy", {"query": "bankruptcy discharge"})


async def test_a_bad_call_raises_a_readable_error():
    async with tool_session() as tools:
        with pytest.raises(RuntimeError):
            await tools.call("dti_ratio", {"monthly_debt": "not a number"})

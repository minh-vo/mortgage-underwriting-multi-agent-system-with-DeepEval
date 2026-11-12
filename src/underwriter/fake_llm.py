"""A deterministic stand-in for Claude, used by the tests. No network, no API key."""

import json
import re
from contextlib import asynccontextmanager

from . import a2a_client, llm
from .agents import critic
from .models import AnalysisDraft, CriticReview, DecisionDraft, SupervisorPick

SECTION_FOR_AREA = {"credit": "1.1", "income": "2.5", "assets": "3.1", "collateral": "4.3"}


def block(text: str, tag: str) -> str:
    """The contents of <tag>...</tag> in a prompt."""
    match = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


@asynccontextmanager
async def _no_client():
    yield None


async def _critic_in_process(payload: dict, url=None, http_client=None) -> dict:
    """Stands in for the A2A call: runs the Critic in this process."""
    return await critic.review(payload["case_id"], payload["application"],
                               payload["metrics"], payload["analyses"])


class FakeLLM:
    """Answers by role. Configure the outcome with the constructor arguments.

    score:         the risk score the Decision Agent returns.
    defect_rounds: one list of defects per Critic call, in order; [] means a clean review.
    fail:          {area: n} makes that specialist fail its next n runs.
    supervisor:    optional function(allowed_moves) -> SupervisorPick, to force a pick.
    """

    def __init__(self, score: int = 20, defect_rounds: list | None = None,
                 fail: dict | None = None, supervisor=None, decision_label: str | None = None):
        self.score = score
        self.defect_rounds = list(defect_rounds or [])
        self.fail = dict(fail or {})
        self.supervisor = supervisor
        self.decision_label = decision_label
        self.calls: list[dict] = []

    def install(self, monkeypatch, fake_critic_service: bool = False) -> "FakeLLM":
        """Replace the Claude calls. With fake_critic_service, also skip the A2A hop."""
        monkeypatch.setattr(llm, "structured", self.structured)
        monkeypatch.setattr(llm, "tool_loop", self.tool_loop)
        monkeypatch.setattr(llm, "use_client", _no_client)
        if fake_critic_service:
            monkeypatch.setattr(a2a_client, "review", _critic_in_process)
        return self

    def prompts(self, agent: str) -> list[str]:
        return [call["user"] for call in self.calls if call["agent"] == agent]

    def _usage(self, agent: str) -> dict:
        return {"agent": agent, "model": "fake", "input_tokens": 100, "output_tokens": 50}

    async def tool_loop(self, system, user, tools, call_tool, schema=None, max_rounds=8,
                        effort="medium", agent=""):
        self.calls.append({"agent": agent, "system": system, "user": user})
        if self.fail.get(agent, 0) > 0:
            self.fail[agent] -= 1
            return llm.ToolLoopResult(status="failed", usage=[self._usage(agent)],
                                      error=f"No final answer after {max_rounds} rounds of tool calls.")
        calls = list(json.loads(block(user, "authoritative_tool_inputs")).items())
        calls.append(("get_policy", {"section": SECTION_FOR_AREA[agent]}))
        trace = []
        for name, args in calls:
            trace.append({"tool": name, "input": args, "output": await call_tool(name, args),
                          "is_error": False})
        reworked = "<rework>" in user
        draft = AnalysisDraft(
            summary=f"{agent} analysis" + (" (reworked)" if reworked else ""),
            findings=[call["output"] for call in trace[:-1]],
            policy_refs=[SECTION_FOR_AREA[agent]], risk_level="LOW", conditions=[],
            recommendation="Proceed.")
        return llm.ToolLoopResult(status="complete", output=draft, text=draft.model_dump_json(),
                                  trace=trace, usage=[self._usage(agent)])

    async def structured(self, system, user, schema, effort="high", agent="", model=None):
        self.calls.append({"agent": agent, "system": system, "user": user})
        if schema is SupervisorPick:
            allowed = json.loads(block(user, "allowed_moves"))
            answer = (self.supervisor(allowed) if self.supervisor else
                      SupervisorPick(action=allowed[0]["action"], areas=allowed[0]["areas"],
                                     reason="First allowed move."))
        elif schema is CriticReview:
            defects = self.defect_rounds.pop(0) if self.defect_rounds else []
            answer = CriticReview(defects=defects, application_risks=["Thin reserves."],
                                  policy_violations=[], summary=f"{len(defects)} defect(s) found.")
        elif schema is DecisionDraft:
            label = self.decision_label or ("APPROVED" if self.score <= 30 else
                                            "CONDITIONAL_APPROVAL" if self.score <= 60 else "DENIED")
            conditions = ["Provide a letter of explanation."] if label == "CONDITIONAL_APPROVAL" else []
            answer = DecisionDraft(risk_score=self.score, decision=label, conditions=conditions,
                                   memo=f"## Decision memo\nRisk score {self.score}: {label}.")
        else:
            raise AssertionError(f"FakeLLM has no answer for {schema.__name__}")
        return answer, [self._usage(agent)]

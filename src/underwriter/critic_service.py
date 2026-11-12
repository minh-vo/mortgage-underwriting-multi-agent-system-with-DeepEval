"""The Critic as an A2A service, in its own process.

Run with: python -m underwriter.critic_service
"""

import json
from urllib.parse import urlparse

import uvicorn
from a2a import types as a2a_types
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from starlette.applications import Starlette

from . import llm
from .agents import critic
from .config import get_settings

JSON = "application/json"
AGENT_NAME = "critic-analyst"


def _part(text: str, media_type: str | None = None) -> a2a_types.Part:
    part = a2a_types.Part()
    part.text = text
    if media_type:
        part.media_type = media_type
    return part


class CriticExecutor(AgentExecutor):
    """Receives {case_id, application, metrics, analyses}; returns the review as a JSON artifact."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await event_queue.enqueue_event(a2a_types.Task(
                id=context.task_id, context_id=context.context_id,
                status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_SUBMITTED)))
        await updater.start_work()
        try:
            payload = json.loads(context.get_user_input() or "{}")
            async with llm.use_client():  # this process's own client, closed after the review
                result = await critic.review(payload["case_id"], payload["application"],
                                             payload["metrics"], payload["analyses"])
            await updater.add_artifact([_part(json.dumps(result), JSON)], name="critic_review")
            await updater.complete()
        except Exception as error:  # the caller gets a readable failure, not a hang
            message = updater.new_agent_message(
                [_part(f"Critic failed: {type(error).__name__}: {error}")])
            await updater.failed(message=message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


def build_app(url: str) -> Starlette:
    card = a2a_types.AgentCard(
        name=AGENT_NAME,
        description="Reviews the quality of the specialist analyses of one mortgage file and "
                    "returns defects, borrower risks and a pass/fail verdict.",
        version="1.0.0",
        supported_interfaces=[a2a_types.AgentInterface(
            url=url.rstrip("/") + "/", protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=a2a_types.AgentCapabilities(streaming=False),
        default_input_modes=[JSON],
        default_output_modes=[JSON],
        skills=[a2a_types.AgentSkill(
            id="critic", name="Underwriting quality review",
            description="Checks the credit, income, asset and collateral analyses against the "
                        "authoritative figures and the policy.",
            tags=["underwriting", "review"], input_modes=[JSON], output_modes=[JSON])],
    )
    handler = DefaultRequestHandler(agent_executor=CriticExecutor(),
                                    task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))


def main() -> None:
    url = get_settings().critic_url
    parsed = urlparse(url)
    uvicorn.run(build_app(url), host=parsed.hostname or "127.0.0.1", port=parsed.port or 9105,
                log_level="warning")


if __name__ == "__main__":
    main()

"""A2A client for the Critic service, and a helper that makes sure the service is up."""

import atexit
import json
import subprocess
import sys
import time
import uuid

import httpx
from a2a import types as a2a_types
from a2a.client import A2ACardResolver, A2AClientError, A2AClientTimeoutError, ClientConfig, ClientFactory

from .config import get_settings

JSON = "application/json"
AGENT_NAME = "critic-analyst"
CARD_PATH = "/.well-known/agent-card.json"
# Longer than the Critic's own worst case (180 s per attempt, 2 attempts).
REVIEW_TIMEOUT = 400.0


class CriticUnavailable(RuntimeError):
    """The Critic service could not be reached or did not return a review."""


def critic_status(url: str | None = None) -> str:
    """"running", "down" (nothing listening) or "other" (something else holds the address)."""
    url = url or get_settings().critic_url
    try:
        response = httpx.get(url.rstrip("/") + CARD_PATH, timeout=3.0)
    except httpx.TransportError:
        return "down"
    try:
        return "running" if response.json().get("name") == AGENT_NAME else "other"
    except ValueError:
        return "other"


def ensure_critic_running(url: str | None = None, wait_seconds: float = 30.0):
    """Start the Critic service as a subprocess unless it already answers. Stops it at exit."""
    url = url or get_settings().critic_url
    status = critic_status(url)
    if status == "running":
        return None
    if status == "other":
        raise CriticUnavailable(f"{url} is in use by something that is not the Critic service. "
                                "Stop it, or set CRITIC_URL to a free port.")
    process = subprocess.Popen([sys.executable, "-m", "underwriter.critic_service"])
    atexit.register(process.terminate)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CriticUnavailable(f"The Critic service exited with code {process.returncode}.")
        if critic_status(url) == "running":
            return process
        time.sleep(0.5)
    process.terminate()
    raise CriticUnavailable(f"The Critic service did not answer at {url} within {wait_seconds:.0f} s.")


def _status_text(status) -> str:
    return " ".join(p.text for p in status.message.parts if p.HasField("text")) or "no detail"


async def review(payload: dict, url: str | None = None,
                 http_client: httpx.AsyncClient | None = None) -> dict:
    """Send {case_id, application, metrics, analyses} to the Critic and return its review."""
    url = url or get_settings().critic_url
    own_client = http_client is None
    http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(REVIEW_TIMEOUT, connect=10.0))
    try:
        card = await A2ACardResolver(http, base_url=url).get_agent_card()
        if "critic" not in [skill.id for skill in card.skills]:
            raise CriticUnavailable(f"The agent at {url} does not offer the 'critic' skill.")
        client = ClientFactory(ClientConfig(httpx_client=http, streaming=False)).create(card)

        message = a2a_types.Message(role=a2a_types.Role.ROLE_USER, message_id=str(uuid.uuid4()),
                                    context_id=f"{payload['case_id']}-{uuid.uuid4().hex[:8]}")
        part = a2a_types.Part()
        part.text, part.media_type = json.dumps(payload), JSON
        message.parts.append(part)
        request = a2a_types.SendMessageRequest(
            message=message,
            configuration=a2a_types.SendMessageConfiguration(accepted_output_modes=[JSON]))

        artifact_text = None
        async for response in client.send_message(request):
            if response.HasField("artifact_update"):
                for item in response.artifact_update.artifact.parts:
                    if item.HasField("text"):
                        artifact_text = item.text
            if not response.HasField("task"):
                continue
            task = response.task
            if task.status.state == a2a_types.TaskState.TASK_STATE_FAILED:
                raise CriticUnavailable(_status_text(task.status))
            if task.status.state == a2a_types.TaskState.TASK_STATE_CANCELED:
                raise CriticUnavailable("The Critic review was cancelled.")
            for artifact in task.artifacts:
                for item in artifact.parts:
                    if item.HasField("text"):
                        artifact_text = item.text
        if artifact_text is None:
            raise CriticUnavailable("The Critic returned no review.")
        return json.loads(artifact_text)
    except (httpx.TimeoutException, A2AClientTimeoutError) as error:
        raise CriticUnavailable(f"The Critic did not answer within {REVIEW_TIMEOUT:.0f} s.") from error
    except (httpx.TransportError, A2AClientError) as error:
        raise CriticUnavailable(
            f"The Critic service is not reachable at {url}. Start it with "
            f"`python -m underwriter.critic_service`. Detail: {error}") from error
    finally:
        if own_client:
            await http.aclose()

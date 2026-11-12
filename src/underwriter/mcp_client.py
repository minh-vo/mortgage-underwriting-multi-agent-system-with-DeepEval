"""MCP client: start the tool server as a subprocess and talk to it over stdio."""

import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ToolSession:
    """One open connection to the tool server."""

    def __init__(self, session: ClientSession):
        self._session = session

    async def anthropic_tools(self) -> list[dict]:
        """The server's tools as Anthropic tool definitions."""
        listed = await self._session.list_tools()
        return [{"name": tool.name, "description": tool.description or "",
                 "input_schema": tool.inputSchema} for tool in listed.tools]

    async def call(self, name: str, args: dict) -> str:
        """Call a tool and return its text. Raises RuntimeError when the tool fails."""
        result = await self._session.call_tool(name, args)
        text = "\n".join(part.text for part in result.content if part.type == "text")
        if result.isError:
            raise RuntimeError(text or f"tool {name} failed")
        return text


@asynccontextmanager
async def tool_session():
    """Open a tool session. One per specialist run: simple, and nothing is shared."""
    server = StdioServerParameters(command=sys.executable, args=["-m", "underwriter.mcp_server"])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield ToolSession(session)

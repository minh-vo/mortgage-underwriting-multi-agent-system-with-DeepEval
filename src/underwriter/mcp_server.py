"""MCP tool server (stdio): the seven calculators plus policy lookup.

Run with: python -m underwriter.mcp_server
"""

from mcp.server.fastmcp import FastMCP

from . import metrics, policy

mcp = FastMCP("underwriting-tools", log_level="WARNING")

for calculator in metrics.CALCULATORS:
    mcp.add_tool(calculator)


@mcp.tool()
def get_policy(section: str) -> str:
    """Return the policy text for a section number such as "2.5", or a whole chapter such as "2".

    Args:
        section: Section number ("2.5") or chapter number ("2").
    """
    return policy.get_section(section)


@mcp.tool()
def search_policy(query: str) -> str:
    """Search the underwriting policy by keywords. Results include their section numbers.

    Args:
        query: A few keywords, for example "large deposit documentation".
    """
    return policy.search(query)


if __name__ == "__main__":
    mcp.run()  # stdio transport

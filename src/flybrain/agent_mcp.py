"""MCP client side: launch our flybrain server over stdio and expose its tools to LangChain.

The agent never imports flybrain.atlas/tools; it only talks to the server over MCP.

Why a persistent session: ``MultiServerMCPClient.get_tools()`` hands out tools that open
a NEW session (= a new server process) for every call. Selection tickets live in the
server process, so a ticket from search_neurons would be dead by the next call. Holding
one ``client.session(...)`` open for the whole conversation keeps the same server alive.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

SERVER_NAME = "flybrain"


@asynccontextmanager
async def mcp_tools(annotations_path: str | Path | None = None):
    """Yield the flybrain tools, all bound to ONE live server session."""
    connection = {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "flybrain.mcp_server"],
    }
    if annotations_path is not None:
        connection["env"] = {"FLYBRAIN_ANNOTATIONS": str(annotations_path)}
    client = MultiServerMCPClient({SERVER_NAME: connection})
    async with client.session(SERVER_NAME) as session:
        yield await load_mcp_tools(session, server_name=SERVER_NAME)

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

import anyio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.exceptions import McpError

SERVER_NAME = "flybrain"


class ServerStoppedError(RuntimeError):
    """The flybrain MCP server died mid-conversation."""


CONNECTION_CLOSED = -32000  # JSON-RPC code the mcp SDK uses for "Connection closed"


def _walk(exc: BaseException | None, seen: set[int]):
    """Yield exc, everything it chains to, and every member of exception groups."""
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        if isinstance(exc, BaseExceptionGroup):
            for sub in exc.exceptions:
                yield from _walk(sub, seen)
        for nxt in (exc.__cause__, exc.__context__):
            yield from _walk(nxt, seen)
        return


def server_gone(exc: BaseException) -> BaseException | None:
    """The transport-level error inside `exc` if the server died, else None.

    Looks for what the mcp/anyio stack raises once the stdio pipe is gone: McpError
    "Connection closed" first, then ClosedResourceError / BrokenResourceError (and plain
    broken pipes) on every later call. Ordinary tool failures (bad args, server-side
    ToolError text) are NOT transport errors and return None.
    """
    for err in _walk(exc, set()):
        if isinstance(err, (anyio.ClosedResourceError, anyio.BrokenResourceError,
                            anyio.EndOfStream, BrokenPipeError, ConnectionResetError)):
            return err
        if isinstance(err, McpError) and err.error.code == CONNECTION_CLOSED:
            return err
    return None


def stopped_error(cause: BaseException) -> ServerStoppedError:
    reason = " ".join(str(cause).split()) or type(cause).__name__
    return ServerStoppedError(f"the flybrain server stopped: {reason}")


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

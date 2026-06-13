# llmmd MCP (lite)

Stdio MCP server with two tools: `rag_search` and `rag_ask`. Both call the running RAG API (`python llmmd.py rag`).

## Setup

```powershell
python llmmd.py setup mcp
python llmmd.py mcp-config
```

Copy the printed `mcpServers.llmmd-rag` block into Cursor or Claude Desktop MCP settings. Set `LLMMD_RAG_BASE_URL` if RAG runs on a non-default host/port.

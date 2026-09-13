"""Tool registry. Add new tools by writing a function and registering it here."""
from app.tools.web_search import web_search
from app.tools.email import read_email

# Registry: tool name -> function(query: str, keys: dict) -> str
TOOLS = {
    "web_search": web_search,
    "read_email": read_email,
}

# Human-readable description injected into the prompt so the model knows what exists.
TOOLS_DESCRIPTION = """You have access to these tools:
- web_search: search the internet for current information. Use for recent events, facts, anything you may not know.
- read_email: read the user's most recent Gmail messages. Use when the user asks about their email.

To use a tool, reply with ONLY a JSON object on a single line, nothing else:
{"tool": "web_search", "query": "your search query"}
or
{"tool": "read_email", "query": "what to look for"}

If you do NOT need a tool, just answer normally in plain text."""

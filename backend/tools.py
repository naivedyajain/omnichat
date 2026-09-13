"""Tools module: Tavily Web Search and Read-Only Gmail IMAP Access."""
import email
from email.header import decode_header
import imaplib
import json
import re
import requests


def _decode_header_value(value):
    if not value:
        return ""
    parts = decode_header(value)
    out = ""
    for text, enc in parts:
        if isinstance(text, bytes):
            out += text.decode(enc or "utf-8", errors="ignore")
        else:
            out += text
    return out


def web_search(query: str, keys: dict) -> str:
    """Execute web search via Tavily API with detailed step-by-step console logging."""
    api_key = keys.get("tavily", "")
    print("[Tavily Log] Initiating web search...")
    print(f"[Tavily Log] Query: '{query}'")
    print(f"[Tavily Log] API Key present: {bool(api_key)}, length: {len(api_key)}")

    if not api_key:
        err = "ERROR: No Tavily API key provided in settings."
        print(f"[Tavily Log] {err}")
        return err

    try:
        print("[Tavily Log] Sending POST request to https://api.tavily.com/search...")
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": 5,
                "include_answer": True,
            },
            timeout=30,
        )
        print(f"[Tavily Log] Response HTTP status code: {resp.status_code}")

        data = resp.json()
        print("[Tavily Log] FULL TAVILY API RESPONSE:")
        print(json.dumps(data, indent=2))

        parts = []
        answer = data.get("answer")
        if answer:
            print(f"[Tavily Log] Extracted summary answer: '{answer}'")
            parts.append("Summary: " + answer)

        results = data.get("results", [])
        print(f"[Tavily Log] Processing {len(results)} search results...")
        for idx, r in enumerate(results, start=1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            print(f"[Tavily Log] Result #{idx}: title='{title}', url='{url}'")
            parts.append(f"- {title}: {content} ({url})")

        output = "\n".join(parts) if parts else "No results found."
        print(f"[Tavily Log] Search parsing complete. Output length: {len(output)} chars.")
        return output

    except Exception as e:
        err_msg = f"ERROR during web search: {repr(e)}"
        print(f"[Tavily Log] {err_msg}")
        return err_msg


def read_email(query: str, keys: dict) -> str:
    """Fetch recent emails over Gmail IMAP SSL (993) with regex credential sanitization."""
    address = keys.get("gmail_address", "")
    raw_pw = keys.get("gmail_app_password", "")

    if address:
        address = re.sub(r'[\s"\'<>]', "", address)

    app_password = re.sub(r"[^A-Za-z0-9]", "", raw_pw) if raw_pw else ""
    print(f"[Gmail Log] Cleaned email: '{address}', password length: {len(app_password)}")

    if not address or not app_password:
        return "⚠️ Gmail Diagnostic: Gmail address or app password not provided in settings."

    if len(app_password) != 16:
        print(f"[Gmail Log] WARNING: App password length is {len(app_password)}, expected 16 characters.")

    imaplib._MAXLINE = 10000000

    try:
        print("[Gmail Log] Connecting to imap.gmail.com:993...")
        mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)

        print("[Gmail Log] Attempting login...")
        mail.login(address, app_password)
        print("[Gmail Log] Login result: SUCCESS")

        print("[Gmail Log] Selecting INBOX (readonly)...")
        status, data = mail.select("INBOX", readonly=True)
        total = int(data[0]) if data and data[0] else 0
        print(f"[Gmail Log] INBOX total messages: {total}")

        if total == 0:
            print("[Gmail Log] INBOX empty. Trying [Gmail]/All Mail...")
            status, data = mail.select('"[Gmail]/All Mail"', readonly=True)
            total = int(data[0]) if data and data[0] else 0
            print(f"[Gmail Log] [Gmail]/All Mail total messages: {total}")

        if total == 0:
            mail.logout()
            return "No emails found."

        start = max(1, total - 9)
        end = total
        print(f"[Gmail Log] Fetching messages sequence range: {start} to {end}")

        out = []
        for num in range(end, start - 1, -1):
            status, msg_data = mail.fetch(str(num), "(BODY.PEEK[])")
            if not msg_data or not msg_data[0]:
                continue

            raw_content = msg_data[0][1]
            if not isinstance(raw_content, bytes):
                continue

            msg = email.message_from_bytes(raw_content)
            subject = _decode_header_value(msg.get("Subject"))
            sender = _decode_header_value(msg.get("From"))

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode(errors="ignore")
                        except Exception:
                            body = ""
                        break
            else:
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode(errors="ignore")
                except Exception:
                    body = ""

            body_snippet = body.strip()[:1000] if body else "(No plain text content)"
            out.append(f"From: {sender}\nSubject: {subject}\n\n{body_snippet}")

        print(f"[Gmail Log] Successfully fetched {len(out)} messages.")
        mail.logout()
        return "\n\n---\n\n".join(out) if out else "No emails found."

    except Exception as e:
        err_msg = repr(e)
        print(f"[Gmail Log] ERROR: {err_msg}")
        return f"⚠️ Gmail Diagnostic: {err_msg}"


TOOLS = {
    "web_search": web_search,
    "read_email": read_email,
}

TOOLS_DESCRIPTION = """You have access to these tools:
- web_search: search the internet for current information. Use for recent events, facts, anything you may not know.
- read_email: read the user's most recent Gmail messages. Use when the user asks about their email.

To use a tool, reply with ONLY a JSON object on a single line, nothing else:
{"tool": "web_search", "query": "your search query"}
or
{"tool": "read_email", "query": "what to look for"}

If you do NOT need a tool, just answer normally in plain text."""
